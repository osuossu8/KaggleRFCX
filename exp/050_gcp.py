# ====================================================
# Directory settings
# ====================================================
import os
import sys
sys.path.append("/home/osuosuossu/KaggleRFCX")

OUTPUT_DIR = 'outputs/exp050_gcp/'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


# ====================================================
# CFG
# ====================================================
class CFG:
    debug=False
    apex=False # True
    num_workers=0 # 8
    model_name= "seresnext50_32x4d"
    model_param = {
        'encoder' : model_name,
        'classes_num' : 24
    }
    duration=10
    period=6
    scheduler='CosineAnnealingLR' # ['ReduceLROnPlateau', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts']
    step_scheduler=True
    epochs=60 # 50
    #factor=0.2 # ReduceLROnPlateau
    #patience=4 # ReduceLROnPlateau
    T_max=10 # CosineAnnealingLR
    T_0=10 # CosineAnnealingWarmRestarts
    lr=1e-3
    min_lr=0. # 1e-6
    batch_size=64 # 24
    weight_decay=1e-6
    gradient_accumulation_steps=1
    max_grad_norm=1000
    alpha=1.0
    mixup_epochs=0 # 40
    p_mixup=0. # 0.5
    p_cutmix=0.
    seed=6718 # 777
    target_size=24
    target_col='target'
    n_fold=5
    trn_fold=[0, 1, 2, 3, 4]
    train=True
    inference=True

if CFG.debug:
    CFG.epochs = 2
    CFG.trn_fold = [0, 1]


# ====================================================
# Library
# ====================================================
import os
import sys
import copy
import gc
import time
import math
import random
import shutil
from pathlib import Path
from contextlib import contextmanager
from collections import defaultdict, Counter

import cv2
import librosa
import audioread
from PIL import Image
import soundfile as sf
from torchlibrosa.augmentation import SpecAugmentation
from torchlibrosa.stft import Spectrogram, LogmelFilterBank

import scipy as sp
import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold

from tqdm.auto import tqdm
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau

from transformers import get_linear_schedule_with_warmup

import audiomentations as A
import kornia.augmentation as K

import timm

import warnings 
warnings.filterwarnings('ignore')

if CFG.apex:
    from apex import amp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ====================================================
# Utils
# ====================================================
def _one_sample_positive_class_precisions(scores, truth):
    num_classes = scores.shape[0]
    pos_class_indices = np.flatnonzero(truth > 0)
    if not len(pos_class_indices):
        return pos_class_indices, np.zeros(0)
    retrieved_classes = np.argsort(scores)[::-1]
    class_rankings = np.zeros(num_classes, dtype=np.int)
    class_rankings[retrieved_classes] = range(num_classes)
    retrieved_class_true = np.zeros(num_classes, dtype=np.bool)
    retrieved_class_true[class_rankings[pos_class_indices]] = True
    retrieved_cumulative_hits = np.cumsum(retrieved_class_true)
    precision_at_hits = (
            retrieved_cumulative_hits[class_rankings[pos_class_indices]] /
            (1 + class_rankings[pos_class_indices].astype(np.float)))
    return pos_class_indices, precision_at_hits


def lwlrap(truth, scores):
    assert truth.shape == scores.shape
    num_samples, num_classes = scores.shape
    precisions_for_samples_by_classes = np.zeros((num_samples, num_classes))
    for sample_num in range(num_samples):
        pos_class_indices, precision_at_hits = _one_sample_positive_class_precisions(scores[sample_num, :], truth[sample_num, :])
        precisions_for_samples_by_classes[sample_num, pos_class_indices] = precision_at_hits
    labels_per_class = np.sum(truth > 0, axis=0)
    weight_per_class = labels_per_class / float(np.sum(labels_per_class))
    per_class_lwlrap = (np.sum(precisions_for_samples_by_classes, axis=0) /
                        np.maximum(1, labels_per_class))
    return per_class_lwlrap, weight_per_class


def get_score(y_true, y_pred):
    """
    y_true = np.array([[1, 0, 0], [0, 0, 1]])
    y_pred = np.array([[0.75, 0.5, 1], [1, 0.2, 0.1]])
    """
    score_class, weight = lwlrap(y_true, y_pred)
    score = (score_class * weight).sum()
    return score


def init_logger(log_file=OUTPUT_DIR+'train.log'):
    from logging import getLogger, INFO, FileHandler,  Formatter,  StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

LOGGER = init_logger()


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_torch(seed=CFG.seed)


# ====================================================
# Data Loading
# ====================================================
traint = pd.read_csv('inputs/train_tp.csv')
trainf = pd.read_csv('inputs/train_fp.csv')
traintpl = pd.read_csv('intermids/pseudo_labeled_additional_train_tp.csv')
trainfpl = pd.read_csv('intermids/pseudo_labeled_additional_train_fp.csv')
threshold = 0.98 # 0.95
traintpl = traintpl[traintpl['max_confidence']>threshold].reset_index(drop=True)
del traintpl['kfold'], traintpl['max_confidence'], traintpl['mean_confidence']
trainfpl = trainfpl[trainfpl['max_confidence']>threshold].reset_index(drop=True)
del trainfpl['kfold'], trainfpl['max_confidence'], trainfpl['mean_confidence']

traint["istp"] = 1
trainf["istp"] = 0
traintpl["istp"] = 1 # 0.5
trainfpl["istp"] = 1

test = pd.read_csv('inputs/sample_submission.csv')
print(traint.shape, trainf.shape, test.shape)

if CFG.debug:
    test = test.head()

PERIOD = CFG.period
TIME = CFG.duration
SR = 48000
FMIN = 20 # 40
FMAX = SR // 2
IMAGE_WIDTH = 456
IMAGE_HEIGHT = 456 #  224 # 320
N_MELS = IMAGE_HEIGHT
HOP_SIZE = 512
WINDOW_SIZE = 512*6

# 各speciesのfmaxとfminを求める
species_fmin = traint.groupby("species_id")["f_min"].agg(min).reset_index()
species_fmax = traint.groupby("species_id")["f_max"].agg(max).reset_index()
species_fmin_fmax = pd.merge(species_fmin, species_fmax, on="species_id")
#print(species_fmin_fmax)

MEL_FREQ = librosa.mel_frequencies(fmin=FMIN, fmax=FMAX, n_mels=IMAGE_HEIGHT)

def search_bin(value):
    n = 0
    for i, v in enumerate(MEL_FREQ):
        if v < value:
            pass
        else:
            n = i - 1
            break
    return n

# mel specに変換した時の座標を求める
# https://akifukka.hatenablog.com/entry/text2speech2
species_fmin_fmax["f_min_mel"] = species_fmin_fmax["f_min"].map(search_bin)
species_fmin_fmax["f_max_mel"] = species_fmin_fmax["f_max"].map(search_bin)

# train_tpにmelの情報をmergeする
species_fmin_fmax['species_id'] = species_fmin_fmax['species_id'].astype(int)
traint['species_id'] = traint['species_id'].astype(int)
trainf['species_id'] = trainf['species_id'].astype(int)
species_fmin_fmax.drop(["f_min", "f_max"], inplace=True, axis=1)
traint = pd.merge(traint, species_fmin_fmax, on="species_id", how="left")
trainf = pd.merge(trainf, species_fmin_fmax, on="species_id", how="left")
traintpl = pd.merge(traintpl, species_fmin_fmax, on="species_id", how="left")
traintpl = pd.merge(traintpl, traint[['recording_id', 'f_min', "f_max"]], on="recording_id", how="left")
trainfpl = pd.merge(trainfpl, traint[['recording_id', 'f_min', "f_max"]], on="recording_id", how="left")

# tpとfpをconcat
train_df = pd.concat([traint, trainf], axis=0).reset_index()
print(train_df.shape)


# ====================================================
# CV split
# ====================================================

# https://www.kaggle.com/ttahara/ranzcr-multi-head-model-training
def multi_label_stratified_group_k_fold(label_arr: np.array, gid_arr: np.array, n_fold: int, seed: int=42):
    """
    create multi-label stratified group kfold indexs.
    reference: https://www.kaggle.com/jakubwasikowski/stratified-group-k-fold-cross-validation
    input:
        label_arr: numpy.ndarray, shape = (n_train, n_class)
            multi-label for each sample's index using multi-hot vectors
        gid_arr: numpy.array, shape = (n_train,)
            group id for each sample's index
        n_fold: int. number of fold.
        seed: random seed.
    output:
        yield indexs array list for each fold's train and validation.
    """
    np.random.seed(seed)
    random.seed(seed)
    start_time = time.time()
    n_train, n_class = label_arr.shape
    gid_unique = sorted(set(gid_arr))
    n_group = len(gid_unique)
    # # aid_arr: (n_train,), indicates alternative id for group id.
    # # generally, group ids are not 0-index and continuous or not integer.
    gid2aid = dict(zip(gid_unique, range(n_group)))
    aid_arr = np.vectorize(lambda x: gid2aid[x])(gid_arr)
    # # count labels by class
    cnts_by_class = label_arr.sum(axis=0)  # (n_class, )
    # # count labels by group id.
    col, row = np.array(sorted(enumerate(aid_arr), key=lambda x: x[1])).T
    cnts_by_group = sp.sparse.coo_matrix(
        (np.ones(len(label_arr)), (row, col))
    ).dot(sp.sparse.coo_matrix(label_arr)).toarray().astype(int)
    del col
    del row
    cnts_by_fold = np.zeros((n_fold, n_class), int)
    groups_by_fold = [[] for fid in range(n_fold)]
    group_and_cnts = list(enumerate(cnts_by_group))  # pair of aid and cnt by group
    np.random.shuffle(group_and_cnts)
    print("finished preparation", time.time() - start_time)
    for aid, cnt_by_g in sorted(group_and_cnts, key=lambda x: -np.std(x[1])):
        best_fold = None
        min_eval = None
        for fid in range(n_fold):
            # # eval assignment.
            cnts_by_fold[fid] += cnt_by_g
            fold_eval = (cnts_by_fold / cnts_by_class).std(axis=0).mean()
            cnts_by_fold[fid] -= cnt_by_g
            if min_eval is None or fold_eval < min_eval:
                min_eval = fold_eval
                best_fold = fid
        cnts_by_fold[best_fold] += cnt_by_g
        groups_by_fold[best_fold].append(aid)
    print("finished assignment.", time.time() - start_time)
    gc.collect()
    idx_arr = np.arange(n_train)
    for fid in range(n_fold):
        val_groups = groups_by_fold[fid]
        val_indexs_bool = np.isin(aid_arr, val_groups)
        train_indexs = idx_arr[~val_indexs_bool]
        val_indexs = idx_arr[val_indexs_bool]
        yield train_indexs, val_indexs

train_df = train_df.drop(columns='index').reset_index()
target_cols = [c for c in range(24)]
# tp split
tp_folds = pd.pivot_table(train_df[train_df['istp']==1][['index', 'species_id']], index='index', columns='species_id', aggfunc=len).fillna(0)
tp_folds = tp_folds.reset_index()
tp_folds = tp_folds.merge(train_df, on='index', how='left')
label_arr = tp_folds[target_cols].values
group_id = tp_folds['recording_id'].values
train_val_indexs = list(multi_label_stratified_group_k_fold(label_arr, group_id, CFG.n_fold, CFG.seed))
tp_folds["kfold"] = -1
for fold_id, (trn_idx, val_idx) in enumerate(train_val_indexs):
    tp_folds.loc[val_idx, "kfold"] = fold_id
# fp split
fp_folds = pd.pivot_table(train_df[train_df['istp']==0][['index', 'species_id']], index='index', columns='species_id', aggfunc=len).fillna(0)
fp_folds = fp_folds.reset_index()
fp_folds = fp_folds.merge(train_df, on='index', how='left')
fp_folds = fp_folds[~fp_folds['recording_id'].isin(tp_folds['recording_id'].unique())].reset_index(drop=True) # tp_foldsに存在するrecording_idは除く
label_arr = fp_folds[target_cols].values
group_id = fp_folds['recording_id'].values
train_val_indexs = list(multi_label_stratified_group_k_fold(label_arr, group_id, CFG.n_fold, CFG.seed))
fp_folds["kfold"] = -1
for fold_id, (trn_idx, val_idx) in enumerate(train_val_indexs):
    fp_folds.loc[val_idx, "kfold"] = fold_id
# merge split
merge_df = pd.concat([tp_folds, fp_folds])
merge_df = merge_df[['recording_id', 'kfold']].groupby('recording_id', as_index=False).mean()
train_df = train_df.merge(merge_df, on="recording_id", how="left")
print(train_df.kfold.value_counts())
train_df.to_csv(OUTPUT_DIR+"folds.csv", index=False)
species_fmin_fmax.to_csv(OUTPUT_DIR+"species_fmin_fmax.csv", index=False)

# add pseudo-labeled tp
traintpl = traintpl.merge(merge_df, on="recording_id", how="left").reset_index()
print(traintpl.kfold.value_counts())
traintpl.to_csv(OUTPUT_DIR+"folds_additional_from_tp.csv", index=False)

# add pseudo-labeled fp
trainfpl = trainfpl.merge(merge_df, on="recording_id", how="left").reset_index()
print(trainfpl.kfold.value_counts())
trainfpl.to_csv(OUTPUT_DIR+"folds_additional_from_fp.csv", index=False)

# ====================================================
# audiomentations
# ====================================================
augmenter = A.Compose([
    A.AddGaussianNoise(min_amplitude=0.01, max_amplitude=0.03, p=0.2),
    # A.PitchShift(min_semitones=-3, max_semitones=3, p=0.2),
    A.Gain(p=0.2)
])


# ====================================================
# Dataset
# ====================================================
def cut_spect(spect, fmin_mel, fmax_mel): # height, width
    spect = spect[fmin_mel:fmax_mel]
    return spect


def do_normalize(img):  #img: (bs, ch, w, h)
    bs, ch, w, h = img.shape
    _img = img.clone()
    _img = _img.view(bs, -1)
    _img -= _img.min(1, keepdim=True)[0]
    _img /= _img.max(1, keepdim=True)[0]
    _img = _img.view(bs, ch, w, h) * 255
    return _img


class AudioDataset(Dataset):
    def __init__(self, df, period=PERIOD, time=TIME,
                 transforms=None, data_path="inputs/train"):
        dfgby = df.groupby("recording_id").agg(lambda x: list(x)).reset_index()
        self.period = period
        self.transforms = transforms
        self.data_path = data_path
        self.time = time

        self.recording_ids = dfgby["recording_id"].values
        self.species_ids = dfgby["species_id"].values
        self.t_mins = dfgby["t_min"].values
        self.t_maxs = dfgby["t_max"].values
        self.f_mins = dfgby["f_min"].values
        self.f_maxs = dfgby["f_max"].values
        self.f_min_mels = dfgby["f_min_mel"].values
        self.f_max_mels = dfgby["f_max_mel"].values
        self.istps = dfgby["istp"].values

    def __len__(self):
        return len(self.recording_ids)

    def __getitem__(self, idx):

        recording_id = self.recording_ids[idx]
        species_id = self.species_ids[idx]
        istp = self.istps[idx]
        t_min, t_max = self.t_mins[idx], self.t_maxs[idx]
        f_min, f_max = self.f_mins[idx], self.f_maxs[idx]
        f_min_mel, f_max_mel = self.f_min_mels[idx], self.f_max_mels[idx]

        # 読み込む
        y, sr = sf.read(f"{self.data_path}/{recording_id}.flac")

        len_y = len(y) # 全フレーム数

        # sampling rate(frame/sec)と取得期間(sec)をかけて必要なフレームを取得
        effective_length = sr * self.time

        rint = np.random.randint(len(t_min))

        # tmin, tmaxをフレーム数に変換
        tmin, tmax = round(sr * t_min[rint]), round(sr * t_max[rint])

        cut_min = max(0, min(tmin - (effective_length - (tmax-tmin)) // 2,
                      min(tmax + (effective_length - (tmax-tmin)) // 2,
                      len_y) - effective_length))
        extra = tmax+(effective_length - (tmax-tmin))//2 - len_y
        lack = tmin - (effective_length - (tmax-tmin)) // 2
        start = cut_min + np.random.randint(0, (self.time-self.period)*sr)
        if extra > 0:
            if tmax-(tmax-tmin)//2-self.period*sr < len_y-self.period*sr:
                start = np.random.randint(tmax-(tmax-tmin)//2-self.period*sr, len_y-self.period*sr)
            else:
                start = np.random.randint(tmax-(tmax-tmin)//2-self.period*sr-(len_y-self.period*sr), len_y-self.period*sr)
        if lack < 0:
            if tmin == 0:
                tmin = 1
            start = cut_min + np.random.randint(0, tmin)
                     

        end = start + self.period * sr
        y = y[start : end]

        if self.transforms:
            # 音声のAugumentation(gaussianノイズとか)が入ってる
            y = self.transforms(samples=y, sample_rate=sr)

        # start(フレーム数)->time(sec)に変換
        # start_timeはeffective_lengthの左端
        start_time = start / sr
        end_time = end / sr

        label = np.zeros(24, dtype='f')
        new_tmins = []
        new_tmaxs = []
        new_fmins = []
        new_fmaxs = []
        new_sids = []
        new_istp = []
        for i in range(len(t_min)):
            # 今回、複数のt_minから選んでいるため、データによってはTP,FPの期間がオーバーラップしている
            if (t_min[i] >= start_time) & (t_max[i] <= end_time):
                if f_min_mel[rint] <= (f_min_mel[i]+f_max_mel[i])/2 <= f_max_mel[rint]:
                    if label[species_id[i]] == 0:
                        label[species_id[i]] = 1 * istp[i]
                    new_tmins.append(t_min[i]-start_time)
                    new_tmaxs.append(t_max[i]-start_time)
                    new_fmins.append(f_min[i])
                    new_fmaxs.append(f_max[i])
                    new_sids.append(species_id[i])
                    new_istp.append(istp[i])
            elif start_time <= ((t_min[i] + t_max[i]) / 2) <= end_time: # bboxの重心がeffective_lengthの中にある
                if f_min_mel[rint] <= (f_min_mel[i]+f_max_mel[i])/2 <= f_max_mel[rint]:
                    if label[species_id[i]] == 0:
                        label[species_id[i]] = 1 * istp[i]
                    new_tmin = 0
                    new_tmax = 0
                    if t_min[i] - start_time < 0:
                        new_tmin = 0
                    else:
                        new_tmin = t_min[i] - start_time
                    if t_max[i] - start_time < 0:
                        new_tmax = 0
                    elif t_max[i] > end_time:
                        new_tmax = end_time - start_time
                    else:
                        new_tmax = t_max[i] - start_time
                    new_tmins.append(new_tmin)
                    new_tmaxs.append(new_tmax)
                    new_fmins.append(f_min[i])
                    new_fmaxs.append(f_max[i])
                    new_sids.append(species_id[i])
                    new_istp.append(istp[i])

        return {
            "wav": torch.tensor(y, dtype=torch.float),
            "target" : torch.tensor(label, dtype=torch.float),
            "id" : recording_id,
            "f_min_mel": f_min_mel[rint],
            "f_max_mel": f_max_mel[rint],
            }


class ValidDataset(Dataset):
    def __init__(self, df, period=PERIOD, transforms=None, data_path="inputs/train"):
        dfgby = df.groupby("recording_id").agg(lambda x: list(x)).reset_index()
        self.period = period
        self.transforms = transforms
        self.data_path = data_path

        self.recording_ids = dfgby["recording_id"].values
        self.species_ids = dfgby["species_id"].values
        self.t_mins = dfgby["t_min"].values
        self.t_maxs = dfgby["t_max"].values
        self.f_mins = dfgby["f_min"].values
        self.f_maxs = dfgby["f_max"].values
        self.f_min_mels = dfgby["f_min_mel"].values
        self.f_max_mels = dfgby["f_max_mel"].values
        self.istps = dfgby["istp"].values

    def __len__(self):
        return len(self.recording_ids)

    def __getitem__(self, idx):
        recording_id = self.recording_ids[idx]
        species_id = self.species_ids[idx]
        istp = self.istps[idx]
        t_min, t_max = self.t_mins[idx], self.t_maxs[idx]
        f_min, f_max = self.f_mins[idx], self.f_maxs[idx]
        f_min_mel, f_max_mel = self.f_min_mels[idx], self.f_max_mels[idx]

        rint = np.random.randint(len(t_min))

        # 読み込む
        y, sr = sf.read(f"{self.data_path}/{recording_id}.flac")

        # tmin, tmaxをフレーム数に変換
        tmin, tmax = round(sr * t_min[rint]), round(sr * t_max[rint])


        len_y = len(y) # 全フレーム数
        # sampling rate(frame/sec)と取得期間(sec)をかけて必要なフレームを取得
        effective_length = sr * self.period  # 6 sec

        start = 0

        start = max(0, min(tmin - (effective_length - (tmax-tmin)) // 2,
                    min(tmax + (effective_length - (tmax-tmin)) // 2,
                    len_y) - effective_length))
        end = start + effective_length
        y = y[start : end]

        start_time = start / sr
        end_time = end / sr

        label = np.zeros(24, dtype='f')
        new_tmins = []
        new_tmaxs = []
        new_fmins = []
        new_fmaxs = []
        new_sids = []
        new_istp = []
        for i in range(len(t_min)):
            # 今回、複数のt_minから選んでいるため、データによってはTP,FPの期間がオーバーラップしている
            if (t_min[i] >= start_time) & (t_max[i] <= end_time):
                if f_min_mel[rint] <= (f_min_mel[i]+f_max_mel[i])/2 <= f_max_mel[rint]:
                    if label[species_id[i]] == 0:
                        label[species_id[i]] = 1 * istp[i]
                    new_tmins.append(t_min[i]-start_time)
                    new_tmaxs.append(t_max[i]-start_time)
                    new_fmins.append(f_min[i])
                    new_fmaxs.append(f_max[i])
                    new_sids.append(species_id[i])
                    new_istp.append(istp[i])
            elif start_time <= ((t_min[i] + t_max[i]) / 2) <= end_time:  # bboxの重心がeffective_lengthの中にある
                if f_min_mel[rint] <= (f_min_mel[i]+f_max_mel[i])/2 <= f_max_mel[rint]:
                    if label[species_id[i]] == 0:
                        label[species_id[i]] = 1 * istp[i]
                    new_tmin = 0
                    new_tmax = 0
                    if t_min[i] - start_time < 0:
                        new_tmin = 0
                    else:
                        new_tmin = t_min[i] - start_time
                    if t_max[i] - start_time < 0:
                        new_tmax = 0
                    elif t_max[i] > end_time:
                        new_tmax = end_time - start_time
                    else:
                        new_tmax = t_max[i] - start_time
                    new_tmins.append(new_tmin)
                    new_tmaxs.append(new_tmax)
                    new_fmins.append(f_min[i])
                    new_fmaxs.append(f_max[i])
                    new_sids.append(species_id[i])
                    new_istp.append(istp[i])
        return {
            "wav": torch.tensor(y, dtype=torch.float),
            "target" : torch.tensor(label, dtype=torch.float),
            "id" : recording_id,
            "f_min_mel": f_min_mel[rint],
            "f_max_mel": f_max_mel[rint],
            }


class TestDataset(Dataset):
    def __init__(self, df, period=PERIOD, transforms=None, data_path="inputs/test"):
        self.period = period
        self.transforms = transforms
        self.data_path = data_path
        self.recording_ids = df["recording_id"].values


    def __len__(self):
        return len(self.recording_ids)

    def __getitem__(self, idx):

        recording_id = self.recording_ids[idx]

        y, sr = sf.read(f"{self.data_path}/{recording_id}.flac")

        len_y = len(y)
        # フレーム数に変換
        effective_length = sr * self.period

        y_ = []
        i = 0
        while i < len_y:
            # インクリメントしていき全部を舐めていく(effective_lengthずつ飛ばしているけど良い？？)
            y__ = y[i:i+effective_length]
            if effective_length > len(y__):
                break
            else:
                y_.append(y__)
                i = i + int(effective_length)

        y = np.stack(y_)  # (effective_length, 2N)

        label = np.zeros(24, dtype='f')
        # y: clip nums, seq -> clip_nums, width, height
        return {
            "wav" : torch.tensor(y, dtype=torch.float),
            "target" : torch.tensor(label, dtype=torch.float),
            "id" : recording_id,
        }


# ====================================================
# Model
# ====================================================
def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


class AudioClassifier(nn.Module):
    def __init__(self, model_name, n_out):
        super(AudioClassifier, self).__init__()

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=80, time_stripes_num=2, 
                                              freq_drop_width=16, freq_stripes_num=2)
        self.net = timm.create_model(model_name, pretrained=True, in_chans=1)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.3)
        if 'efficientnet' in model_name: 
            n_features = self.net.classifier.in_features
        else:
            n_features = self.net.fc.in_features
        self.net_classifier = nn.Linear(n_features, n_out)
        self.init_weight()

        # korrniaのrandom cropはh,wを想定しているため注意
        self.transform = nn.Sequential(K.RandomHorizontalFlip(p=0.1),
                                       #K.GaussianBlur(7, p=0.5),
                                       #K.RandomCrop((round(IMAGE_HEIGHT*0.7), round(IMAGE_WIDTH*0.7)),p=0.3)
                                       )

    def init_weight(self):
        init_layer(self.net_classifier)
        
    def forward(self, x, f_min_mels, f_max_mels, train=True, test=False):  # x: (bs, 1, w, h)
        # f_min_melとf_max_melによってカットする
        bs, ch, w, h = x.shape

        x = x.reshape(bs*w, -1)
        bsw = bs*w
        spects = []
        fi = 0
        if test:
            for ii, i in enumerate(range(bsw)[::w]):
                spect = x[i:i+w]  # torch (w, h)
                for f_min, f_max in zip(f_min_mels, f_max_mels):
                    _spect = cut_spect(spect.transpose(0,1), f_min, f_max).transpose(0,1)  # out:torch (w, h)

                    # resizeする.
                    _spect = torch.unsqueeze(_spect, 0)
                    _spect = torch.unsqueeze(_spect, 0) # torch(1,1,w,h)
                    _spect = F.interpolate(_spect, (IMAGE_WIDTH, IMAGE_HEIGHT),
                                          mode='bilinear',
                                          align_corners=False)  # out: torch (1, 1, w, h)
                    _spect = torch.squeeze(_spect,0)  #out: torch (1, w, h)
                    spects.append(_spect)
            x = torch.stack(spects)  # torch (bs, 1, w, h)  bs=24*bs*10
        else:
            for ii, i in enumerate(range(bsw)[::w]):
                spect = x[i:i+w]  # torch (w, h)
                f_min = f_min_mels[fi]
                f_max = f_max_mels[fi]
                spect = cut_spect(spect.transpose(0,1), f_min, f_max).transpose(0,1)  # out:torch (w, h)

                # resizeする.
                spect = torch.unsqueeze(spect, 0)
                spect = torch.unsqueeze(spect, 0) # torch(1,1,w,h)
                spect = F.interpolate(spect, (IMAGE_WIDTH, IMAGE_HEIGHT),
                                      mode='bilinear',
                                      align_corners=False)  # out: torch (1, 1, w, h)
                if train:
                    spect = self.transform(spect.transpose(2,3))  # out: torch(1,1,h,w)
                    spect = spect.transpose(2,3)  # out: torch(1,1,w,h)
                spect = torch.squeeze(spect,0)  # torch (1, w, h)
                spects.append(spect)
                fi += 1
            x = torch.stack(spects)  # torch (bs, 1, w, h)
        x = do_normalize(x)
        if train:
            x = self.spec_augmenter(x)

        # x = x.expand(x.shape[0], 3, x.shape[2], x.shape[3])  # channel分複製

        # Output shape (batch size, channels, time, frequency)
        x = self.net.forward_features(x)
        x = self.avg_pool(x).flatten(1)
        x = self.dropout1(x)
        x = self.net_classifier(x)
        return x


# ====================================================
# Loss
# ====================================================
def f1_loss(y_true, y_pred, is_training=False, epsilon=1e-7) -> torch.Tensor:
    '''
    Calculate F1 score. Can work with gpu tensors
    The original implmentation is written by Michal Haltuf on Kaggle.
    Returns
    -------
    torch.Tensor
        `ndim` == 1. 0 <= val <= 1
    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    '''

    y_pred = y_pred > 0.5

    tp = (y_true * y_pred).sum()
    tn = ((1 - y_true) * (1 - y_pred))
    fp = ((1 - y_true) * y_pred).sum()
    fn = (y_true * (1 - y_pred)).sum()

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2* (precision*recall) / (precision + recall + epsilon)
    return f1


# https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/213075
class BCEFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, targets):
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(preds, targets)
        probas = torch.sigmoid(preds)
        loss = targets * self.alpha * (1. - probas)**self.gamma * bce_loss + (1. - targets) * probas**self.gamma * bce_loss
        loss = loss.mean()
        return loss

# ====================================================
# Training helper functions
# ====================================================
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MetricMeter(object):
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.y_true = []
        self.y_pred = []
    
    def update(self, y_true, y_pred):
        self.y_true.extend(y_true.cpu().detach().numpy().tolist())
        self.y_pred.extend(torch.sigmoid(y_pred).cpu().detach().numpy().tolist())

    @property
    def avg(self):
        self.f1score = f1_loss(np.array(self.y_true), np.array(self.y_pred))
        score_class, weight = lwlrap(np.array(self.y_true), np.array(self.y_pred))
        self.score = (score_class * weight).sum()
        return {
            "lwlrap" : self.score,
            "f1score": self.f1score
        }


def get_mixup_indices(bs, f_min_mels, f_max_mels):
    indices_matrix = np.zeros((bs, bs))
    for img1_idx in range(bs):
        for img2_idx in range(bs):
            if img1_idx != img2_idx:
                mix_flag = (f_min_mels[img2_idx] >= f_min_mels[img1_idx]) & (f_max_mels[img2_idx] <= f_max_mels[img1_idx])
                if mix_flag:
                    indices_matrix[img1_idx, img2_idx] = 1
                    break # img1に対してmixupするimg2（img1の周波数帯に存在するもの）が1つ見つかり次第終了
    indices = np.arange(bs)
    indices_matrix_1 = np.where(indices_matrix==1)
    for i, j in zip(indices_matrix_1[0], indices_matrix_1[1]):
        if i in range(bs):
            indices[i] = j
        else:
            indices[i] = i
    return indices


def mixup(data, targets, f_min_mels, f_max_mels, alpha=1.0):
    bs = data.size(0)
    indices = get_mixup_indices(bs, f_min_mels, f_max_mels)
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]
    #lam = np.random.beta(alpha, alpha)
    lam = 0.5
    data = data * lam + shuffled_data * (1 - lam)
    targets = targets * lam + shuffled_targets * (1 - lam)
    return data, targets


def train_epoch(model, spectrogram_extractor, logmel_extractor, loader, 
                criterion, optimizer, scheduler, epoch, device, p_mixup,
                normalize=True, resize=True, spec_aug=True):
    losses = AverageMeter()
    scores = MetricMeter()
    model.train()
    t = tqdm(loader)
    for i, sample in enumerate(t):
        x = sample['wav'].to(device) #(bs, seq)
        target = sample['target'].to(device)
        f_min_mels = sample["f_min_mel"]
        f_max_mels = sample["f_max_mel"]
        x = spectrogram_extractor(x) # (batch_size, 1, time_steps, freq_bins)
        x = logmel_extractor(x)

        #output = model(x, f_min_mels, f_max_mels, train=True)
        #loss = criterion(output, target)
        
        if np.random.rand(1) < p_mixup:
            # mixup
            mix_x, mix_target = mixup(x, target, f_min_mels, f_max_mels)
            output = model(mix_x, f_min_mels, f_max_mels, train=True)
            loss = criterion(output, mix_target)
        else:
            output = model(x, f_min_mels, f_max_mels, train=True)
            loss = criterion(output, target)
        
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        if CFG.apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
        if (i + 1) % CFG.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        if CFG.step_scheduler:
            scheduler.step()
        bs = x.size(0)
        scores.update(target, output)
        losses.update(loss.item(), bs)
        t.set_description(f"Train E:{epoch} - Loss{losses.avg:0.4f}")
    t.close()
    return scores.avg, losses.avg


def valid_epoch(model, spectrogram_extractor, logmel_extractor, 
                loader, criterion, epoch, device):
    losses = AverageMeter()
    scores = MetricMeter()
    model.eval()
    with torch.no_grad():
        t = tqdm(loader)
        for i, sample in enumerate(t):
            x = sample['wav'].to(device)  # (bs, seq)
            target = sample['target'].to(device)
            f_min_mels = sample["f_min_mel"]
            f_max_mels = sample["f_max_mel"]
            x = spectrogram_extractor(x) # (batch_size, 1, time_steps, freq_bins)
            x = logmel_extractor(x)
            bs = x.size(0)
            output = model(x, f_min_mels, f_max_mels, train=False)
            #output = output.reshape(bs, 24, -1)  #(bs, 24, 24) batchsize,
            #output, _ = torch.max(output, dim=1)
            loss = criterion(output, target)
            scores.update(target, output)
            losses.update(loss.item(), bs)
            t.set_description(f"Valid E:{epoch} - Loss:{losses.avg:0.4f}")
    t.close()
    return scores.avg, losses.avg


def test_epoch(model, spectrogram_extractor, logmel_extractor, loader,
               f_min_mels, f_max_mels, device, normalize=True, resize=True):
    model.eval()
    pred_list = []
    id_list = []
    with torch.no_grad():
        t = tqdm(loader)
        for i, sample in enumerate(t):
            x = sample["wav"].to(device)
            bs, c, seq = x.shape
            x = x.reshape(bs*c, seq)
            x = spectrogram_extractor(x)
            x = logmel_extractor(x)
            id = sample["id"]
            output = torch.sigmoid(model(x, f_min_mels, f_max_mels, train=False, test=True))
            output = output.reshape(bs, c*24, -1)
            output, _ = torch.max(output, dim=1)
            output = output.cpu().detach().numpy().tolist()
            pred_list.extend(output)
            id_list.extend(id)
    return pred_list, id_list


def get_valid_all_clip_result(fold):
    # Load Data
    train_df = pd.read_csv(OUTPUT_DIR+'folds.csv')
    train_df = train_df[train_df["istp"]==1].reset_index(drop=True)
    species_fmin_fmax = pd.read_csv(OUTPUT_DIR+"species_fmin_fmax.csv")
    f_min_mels = torch.tensor(species_fmin_fmax["f_min_mel"].values, dtype=torch.int)
    f_max_mels = torch.tensor(species_fmin_fmax["f_max_mel"].values, dtype=torch.int)
    # Load model
    model = AudioClassifier(CFG.model_param["encoder"], CFG.model_param["classes_num"])
    model.load_state_dict(torch.load(OUTPUT_DIR+f'fold-{fold}.bin'))
    model = model.to(device)
    # Get valid
    valid_fold = train_df[train_df.kfold == fold].reset_index(drop=True)
    test_dataset = TestDataset(
        df=valid_fold,
        period=CFG.period,
        transforms=None,
        data_path="inputs/train",
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=CFG.batch_size//64,
        shuffle=False,
        drop_last=False,
        num_workers=CFG.num_workers
    )
    window = 'hann'
    center = True
    pad_mode = 'reflect'
    ref = 1.0
    amin = 1e-10
    top_db = None
    spectrogram_extractor = Spectrogram(n_fft=WINDOW_SIZE, hop_length=HOP_SIZE, 
                                       win_length=WINDOW_SIZE, window=window,
                                       center=center, pad_mode=pad_mode, 
                                       freeze_parameters=True).to(device)
    logmel_extractor = LogmelFilterBank(sr=SR, n_fft=WINDOW_SIZE, 
                                        n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
                                        ref=ref, amin=amin, top_db=top_db, 
                                        freeze_parameters=True).to(device)
    test_pred, ids = test_epoch(model, spectrogram_extractor, logmel_extractor, test_loader,
                                f_min_mels, f_max_mels, device, resize=True)
    test_pred_df = pd.DataFrame({
        "recording_id" : valid_fold.recording_id.values
    })
    test_pred_df["kfold"] = fold
    test_pred_df[[f's{i}' for i in range(24)]] = test_pred
    return test_pred_df


def inference(fold):
    # Load Data
    sub_df = pd.read_csv("inputs/sample_submission.csv")
    species_fmin_fmax = pd.read_csv(OUTPUT_DIR+"species_fmin_fmax.csv")
    f_min_mels = torch.tensor(species_fmin_fmax["f_min_mel"].values, dtype=torch.int)
    f_max_mels = torch.tensor(species_fmin_fmax["f_max_mel"].values, dtype=torch.int)
    # Load model
    model = AudioClassifier(CFG.model_param["encoder"], CFG.model_param["classes_num"])
    model.load_state_dict(torch.load(OUTPUT_DIR+f'fold-{fold}.bin'))
    model = model.to(device)
    # Get valid
    test_dataset = TestDataset(
        df=sub_df,
        period=CFG.period,
        transforms=None,
        data_path="inputs/test",
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=CFG.batch_size//64,
        shuffle=False,
        drop_last=False,
        num_workers=CFG.num_workers
    )
    window = 'hann'
    center = True
    pad_mode = 'reflect'
    ref = 1.0
    amin = 1e-10
    top_db = None
    spectrogram_extractor = Spectrogram(n_fft=WINDOW_SIZE, hop_length=HOP_SIZE,
                                       win_length=WINDOW_SIZE, window=window,
                                       center=center, pad_mode=pad_mode,
                                       freeze_parameters=True).to(device)
    logmel_extractor = LogmelFilterBank(sr=SR, n_fft=WINDOW_SIZE,
                                        n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
                                        ref=ref, amin=amin, top_db=top_db,
                                        freeze_parameters=True).to(device)
    test_pred, ids = test_epoch(model, spectrogram_extractor, logmel_extractor, test_loader,
                                f_min_mels, f_max_mels, device, resize=True)
    test_pred_df = pd.DataFrame({
        "recording_id" : sub_df.recording_id.values
    })
    test_pred_df["kfold"] = fold
    test_pred_df[[f's{i}' for i in range(24)]] = test_pred
    return test_pred_df

# ====================================================
# Train loop
# ====================================================
def train_loop(fold):
    LOGGER.info(f"========== fold: {fold} training ==========")
    train_df = pd.read_csv(OUTPUT_DIR+'folds.csv')
    if CFG.debug:
        train_df = train_df.sample(n=1000, random_state=42)
    traintpl = pd.read_csv(OUTPUT_DIR+"folds_additional_from_tp.csv")
    trainfpl = pd.read_csv(OUTPUT_DIR+"folds_additional_from_fp.csv")
    train_fold_add_fromt = traintpl[traintpl.kfold != fold]
    train_fold_add_fromf = traintpl[trainfpl.kfold != fold]

    train_fold = train_df[train_df.kfold != fold]
    valid_fold = train_df[train_df.kfold == fold]

    train_fold = pd.concat([train_fold, train_fold_add_fromt, train_fold_add_fromf], axis=0)

    train_dataset = AudioDataset(
        df=train_fold,
        period=CFG.period,
        time=CFG.duration,
        transforms=augmenter,
        data_path="inputs/train",
    )
    valid_dataset = ValidDataset(
        df=valid_fold,
        period=CFG.period,
        transforms=None,
        data_path="inputs/train"
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=CFG.num_workers,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=CFG.batch_size//4,
        shuffle=False,
        drop_last=False,
        num_workers=CFG.num_workers
    )
    window = 'hann'
    center = True
    pad_mode = 'reflect'
    ref = 1.0
    amin = 1e-10
    top_db = None
    spectrogram_extractor = Spectrogram(n_fft=WINDOW_SIZE, hop_length=HOP_SIZE, 
                                        win_length=WINDOW_SIZE, window=window,
                                        center=center, pad_mode=pad_mode, 
                                        freeze_parameters=True).to(device)
    logmel_extractor = LogmelFilterBank(sr=SR, n_fft=WINDOW_SIZE, 
                                        n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
                                        ref=ref, amin=amin, top_db=top_db, 
                                        freeze_parameters=True).to(device)

    # ====================================================
    # scheduler
    # ====================================================
    def get_scheduler(optimizer):
        if CFG.scheduler=='ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=CFG.factor, patience=CFG.patience, verbose=True, eps=CFG.min_lr)
        elif CFG.scheduler=='CosineAnnealingLR':
            scheduler = CosineAnnealingLR(optimizer, T_max=CFG.T_max, eta_min=CFG.min_lr, last_epoch=-1)
        elif CFG.scheduler=='CosineAnnealingWarmRestarts':
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=CFG.T_0, T_mult=1, eta_min=CFG.min_lr, last_epoch=-1)
        return scheduler

    # ====================================================
    # model & optimizer
    # ====================================================
    model = AudioClassifier(CFG.model_param["encoder"], CFG.model_param["classes_num"])
    model = model.to(device)

    optimizer = Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay, amsgrad=False)
    scheduler = get_scheduler(optimizer)
    #optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr)
    #num_train_steps = int(len(train_loader) * CFG.epochs)
    #num_warmup_steps = int(0.1 * CFG.epochs * len(train_loader))
    #scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)

    # ====================================================
    # apex
    # ====================================================
    if CFG.apex:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)

    #criterion = nn.BCEWithLogitsLoss()
    criterion = BCEFocalLoss()

    best_score = -np.inf
    
    for epoch in range(CFG.epochs):

        if epoch < CFG.mixup_epochs:
            p_mixup = CFG.p_mixup
        else:
            p_mixup = 0.

        start_time = time.time()
        
        # train
        train_avg, train_loss = train_epoch(model, spectrogram_extractor, logmel_extractor, train_loader, 
                                            criterion, optimizer, scheduler, epoch, device, p_mixup, spec_aug=True)
        
        # valid
        valid_avg, valid_loss = valid_epoch(model, spectrogram_extractor, logmel_extractor, valid_loader,
                                            criterion, epoch, device)
        
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(valid_loss)
        elif isinstance(scheduler, CosineAnnealingLR):
            scheduler.step()
        elif isinstance(scheduler, CosineAnnealingWarmRestarts):
            scheduler.step()

        elapsed = time.time() - start_time
        
        LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {train_loss:.5f}  avg_val_loss: {valid_loss:.5f}  time: {elapsed:.0f}s')
        LOGGER.info(f"Epoch {epoch+1} - train_LWLRAP:{train_avg['lwlrap']:0.5f}  valid_LWLRAP:{valid_avg['lwlrap']:0.5f}")
        LOGGER.info(f"Epoch {epoch+1} - train_F1:{train_avg['f1score']:0.5f}  valid_F1:{valid_avg['f1score']:0.5f}")
        
        if valid_avg['f1score'] > best_score:
            LOGGER.info(f">>>>>>>> Model Improved From {best_score} ----> {valid_avg['f1score']}")
            torch.save(model.state_dict(), OUTPUT_DIR+f'fold-{fold}.bin')
            best_score = valid_avg['f1score']


def get_master_df():
    df = pd.read_csv("inputs/train_tp.csv").sort_values("recording_id")
    df['species_ids'] = df['species_id'].astype(str)
    label_dict = {}
    for recording_id, tmp in df.groupby(['recording_id']):
        label_dict[recording_id] = ' '.join(sorted(set(tmp['species_ids'].values)))
    output = pd.DataFrame({'recording_id': df['recording_id'].unique()})
    output['species_ids'] = output['recording_id'].map(label_dict)
    y_true = np.zeros((len(output), 24))
    for i, species in enumerate(output['species_ids'].values):
        for s in species.split():
            y_true[i, int(s)] = 1
    output[[f'true_s{i}' for i in range(24)]] = y_true
    return output.reset_index(drop=True)


def get_result(oof_df):
    y_true = np.zeros((len(oof_df), 24))
    for i, species in enumerate(oof_df['species_ids'].values):
        for s in species.split():
            y_true[i, int(s)] = 1
    preds = oof_df[[f's{i}' for i in range(24)]].values
    score = get_score(y_true, preds)
    LOGGER.info(f'LWLRAP Score: {score:<.5f}')


# ====================================================
# main
# ====================================================
def main():
    if CFG.train:
        master_df = get_master_df()
        # train 
        oof_df = pd.DataFrame()
        for fold in range(CFG.n_fold):
            if fold in CFG.trn_fold:
                train_loop(fold)
                _oof_df = get_valid_all_clip_result(fold)
                _oof_df = _oof_df.merge(master_df, on='recording_id', how='left')
                oof_df = pd.concat([oof_df, _oof_df])
                LOGGER.info(f"========== fold: {fold} result ==========")
                get_result(_oof_df)
        # CV result
        LOGGER.info(f"========== CV ==========")
        get_result(oof_df)
        # save result
        oof_df.to_csv(OUTPUT_DIR+'oof_df.csv', index=False)

    if CFG.inference:
        # inference
        LOGGER.info(f"========== inference ==========")
        submission = pd.DataFrame()
        for fold in range(CFG.n_fold):
            if fold in CFG.trn_fold:
                sub = inference(fold)
                submission = pd.concat([submission, sub])
        print(f'raw_submission: {submission.shape}')
        submission.to_csv(OUTPUT_DIR+f"raw_submission.csv", index=False)
        sub = submission.groupby("recording_id", as_index=False).mean()
        output_cols = ['recording_id'] + [f's{i}' for i in range(24)]
        print(f'raw_submission: {sub.shape}')
        sub[output_cols].to_csv(OUTPUT_DIR+f"submission.csv", index=False)
        LOGGER.info(f"========== submission ==========")
        LOGGER.info(sub[output_cols].head())


if __name__ == '__main__':
    main()
