import audiomentations as AA
import os, glob, random, time, sys
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import librosa, librosa.display
import soundfile as sf

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from functools import partial
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from transformers import get_linear_schedule_with_warmup, AdamW
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation

import timm
from timm.models.efficientnet import tf_efficientnet_b0_ns

import warnings
warnings.simplefilter('ignore', FutureWarning)

from tqdm import tqdm
tqdm.pandas()

sys.path.append("/root/workspace/KaggleRFCX")
from configs import config as CFG
from src.machine_learning_util import trace, seed_everything, to_pickle, unpickle
from src.competition_util import AudioSEDModel, AudioSEDShortModel
from src.augmentations import train_audio_transform
from src.datasets import SedDatasetV2, SedDatasetV3, SedDatasetV5, SedDatasetTest, SedDatasetTestWithTTA
from src.datasets2 import SedDatasetV7, SedDatasetV8
from src.engine import train_epoch, valid_epoch, test_epoch, lwlrap
from src.losses import PANNsLoss, FocalLoss, PANNsWithFocalLoss, ClassWeightedPANNsLoss


train_audio_transform_v2 = AA.Compose([
    AA.AddGaussianSNR(p=0.5)
])


def valid_preds(loader, model, device):
    model.eval()
    targets = []
    preds = []
    with torch.no_grad():
        t = tqdm(loader)
        for i, sample in enumerate(t):
            input = sample['image'].to(device)
            target = sample['target'].to(device)
            output = model(input)

            preds.append(torch.sigmoid(torch.max(output['framewise_output'], dim=1)[0]).cpu().detach().numpy())
            targets.append(target.cpu().detach().numpy())
        t.close()

    targets = np.concatenate(targets)
    preds = np.concatenate(preds)
    return targets, preds


def main(fold):
    seed_everything(args.seed)

    args.fold = fold
    args.save_path = os.path.join(args.output_dir, args.exp_name)
    # os.makedirs(args.save_path, exist_ok=True)

    train_df = pd.read_csv(args.train_csv)
    # sub_df = pd.read_csv(args.sub_csv)

    valid_fold = train_df[train_df.kfold == fold]
    valid_dataset = SedDatasetV8(
        df = valid_fold,
        period=args.period,
        stride=args.stride,
        audio_transform=None,
        wave_form_mix_up_ratio=None,
        data_path=args.train_data_path,
        mode="valid"
    )

    #test_dataset = SedDatasetTestWithTTA(
    #    df = sub_df,
    #    period=args.period,
    #    stride=args.stride,
    #    audio_transform=train_audio_transform_v2,
    #    wave_form_mix_up_ratio=None,
    #    tta=args.num_tta,
    #    data_path=args.test_data_path,
    #    mode="test"
    #)

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers
    )

    #test_loader = torch.utils.data.DataLoader(
    #    test_dataset,
    #    batch_size=args.batch_size,
    #    shuffle=False,
    #    drop_last=False,
    #    num_workers=args.num_workers
    #)

    model = AudioSEDModel(**args.model_param)
    model = model.to(args.device)
    model.load_state_dict(torch.load(os.path.join(args.save_path, f'fold-{args.fold}.bin'), map_location=args.device))

    val_ys, val_preds = valid_preds(valid_loader, model, args.device)
    
    #target_cols = sub_df.columns[1:].values.tolist()
    #test_pred, ids = test_epoch(args, model, test_loader)
    #tmp = pd.DataFrame()
    #tmp['recording_id'] = ids
    #tmp[target_cols] = test_pred
    #test_pred_df = tmp.groupby('recording_id')[target_cols].mean().reset_index()
    #test_pred_df.to_csv(os.path.join(args.save_path, f"fold-{args.fold}-submission.csv"), index=False)
    #print(os.path.join(args.save_path, f"fold-{args.fold}-submission.csv"))
    
    return val_ys, val_preds


class args:
    exp_name = "EXP040"
    model_param = {
        'encoder' : 'tf_efficientnet_b0_ns',
        'sample_rate': 48000,
        'window_size' : 512 * 2,
        'hop_size' : 345 * 2,
        'mel_bins' : 128,
        'fmin' : 20,
        'fmax' : 18000,
        'classes_num' : 24
    }
    wave_form_mix_up_ratio = 0.6 
    period = 10
    stride = 10
    seed = CFG.SEED
    batch_size = 16
    num_workers = 0
    num_tta = 5

    device = CFG.DEVICE
    train_csv = CFG.TRAIN_FOLDS_PATH
    sub_csv = CFG.SUBMISSION_PATH
    output_dir = "weights"
    train_data_path = CFG.TRAIN_IMG_PATH
    test_data_path = CFG.TEST_IMG_PATH


targets = []
preds = []
for use_fold in range(5):
    tgt, prd = main(fold=use_fold)
    targets.append(tgt)
    preds.append(prd)
targets = np.concatenate(targets)
preds = np.concatenate(preds)

per_class_lwlrap, weight_per_class = lwlrap(targets, preds)
lrap = 0
for (p, w) in zip(per_class_lwlrap, weight_per_class):
    lrap += p * w

final_result = []
final_result.append([lrap]+per_class_lwlrap.tolist())
col_name = ['lrap']+ [f's{i}' for i in range(24)]

result_df = pd.DataFrame(final_result, columns=col_name, index=[args.exp_name])
result_df['LB'] = ['---']
print(result_df.T)


