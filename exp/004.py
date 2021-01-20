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
from src.competition_util import AudioSEDModel
from src.augmentations import train_audio_transform
from src.datasets import SedDatasetV2, SedDatasetTest
from src.engine import train_epoch, valid_epoch, test_epoch
from src.losses import PANNsLoss


def inference(fold):
    seed_everything(args.seed)

    args.fold = fold
    args.save_path = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(args.save_path, exist_ok=True)
        
    sub_df = pd.read_csv(args.sub_csv)

    test_dataset = SedDatasetTest(
        df = sub_df,
        period=args.period,
        stride=5,
        audio_transform=train_audio_transform, # None,
        wave_form_mix_up_ratio=None,
        tta=args.num_tta,
        data_path=args.test_data_path,
        mode="test"
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers
    )

    model = AudioSEDModel(**args.model_param)
    
    # model.load_state_dict(torch.load(os.path.join(args.save_path, f'fold-{args.fold}.bin'), map_location=args.device))
    model.load_state_dict(torch.load(os.path.join(args.pretrain_weights), map_location=args.device))
    model = model.to(args.device)

    target_cols = sub_df.columns[1:].values.tolist()
    test_pred, ids = test_epoch(args, model, test_loader)
    print(np.array(test_pred).shape)
 
    tmp = pd.DataFrame()
    tmp['recording_id'] = ids
    tmp[target_cols] = test_pred
    test_pred_df = tmp.groupby('recording_id')[target_cols].mean().reset_index()

    test_pred_df.to_csv(os.path.join(args.save_path, f"fold-{args.fold}-submission.csv"), index=False)

    print(test_pred_df.shape)
    print(test_pred_df.head())

    print(os.path.join(args.save_path, f"fold-{args.fold}-submission.csv"))


class args:
    DEBUG = False
    PRETRAIN_WITH_NOSIY = False

    exp_name = "SED_E0_5F_BASE"
    pretrain_weights = None
    model_param = {
        'encoder' : 'tf_efficientnet_b0_ns',
        'sample_rate': 48000,
        'window_size' : 512 * 2, # 512 * 2
        'hop_size' : 345 * 2, # 320
        'mel_bins' : 128, # 60
        'fmin' : 20,
        'fmax' : 48000 // 2,
        'classes_num' : 24
    }
    wave_form_mix_up_ratio = None # 0.9
    period = 10
    seed = CFG.SEED
    start_epcoh = 0 
    epochs = 50
    lr = 1e-3
    batch_size = 16
    num_workers = 0
    early_stop = 5 # 15
    step_scheduler = True
    epoch_scheduler = False
    num_tta = 10

    device = CFG.DEVICE
    train_csv = CFG.TRAIN_FOLDS_PATH
    train_noisy_csv = CFG.TRAIN_FOLDS_NOISY_PATH
    sub_csv = CFG.SUBMISSION_PATH
    output_dir = "weights_with_tta"
    train_data_path = CFG.TRAIN_IMG_PATH
    test_data_path = CFG.TEST_IMG_PATH


for use_fold in range(5):
    with trace(f'inference {use_fold}'):
        args.pretrain_weights = f"weights_exp003/SED_E0_5F_BASE/fold-{use_fold}.bin"
        print(args.pretrain_weights)
        inference(fold=use_fold)


