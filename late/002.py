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
from src.competition_util import AudioSEDModel
from src.augmentations import train_audio_transform
from src.datasets import SedDatasetV2, SedDatasetTest
from src.datasetslate import SedDataset5th2nd
from src.engine import train_epoch, valid_epoch, test_epoch
from src.losses import PANNsLoss, FocalLoss, FocalLoss5th, LSEPLoss


train_audio_transform_v2 = AA.Compose([
    AA.AddGaussianSNR(p=0.5),
#    AA.PitchShift(min_semitones=-0.5, max_semitones=0.5, p=0.1),
#    AA.Gain(p=0.2)
])


def main(fold):
    seed_everything(args.seed)

    args.fold = fold
    args.save_path = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(args.save_path, exist_ok=True)

    train_df = pd.read_csv(args.train_csv)
    train_noisy = pd.read_csv(args.train_noisy_csv)        
    train_df['is_tp'] = 1
    train_noisy['is_tp'] = -1
    train_noisy_fold = train_noisy[train_noisy.kfold != fold]

    sub_df = pd.read_csv(args.sub_csv)
    train_fold = train_df[train_df.kfold != fold]
    valid_fold = train_df[train_df.kfold == fold]

    train_fold = pd.concat([train_fold, train_noisy_fold], 0).reset_index(drop=True)

    train_dataset = SedDataset5th2nd(
        df = train_fold,
        period=args.period,
        audio_transform=None, # train_audio_transform_v2,
        wave_form_mix_up_ratio=args.wave_form_mix_up_ratio,
        data_path=args.train_data_path,
        mode="train"
    )

    valid_dataset = SedDatasetV2(
        df = valid_fold,
        period=args.period,
        stride=10,
        audio_transform=None,
        wave_form_mix_up_ratio=None,
        data_path=args.train_data_path,
        mode="valid"
    )

    test_dataset = SedDatasetTest(
        df = sub_df,
        period=args.period,
        stride=10,
        data_path=args.test_data_path
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers
    )

    model = AudioSEDModel(**args.model_param)
    model = model.to(args.device)

    if args.pretrain_weights:
        print("---------------------loading pretrain weights")
        model.load_state_dict(torch.load(args.pretrain_weights, map_location=args.device), strict=False)
        model = model.to(args.device)

    criterion = FocalLoss5th(stage=2) # LSEPLoss() # PANNsLoss() #BCEWithLogitsLoss() #MaskedBCEWithLogitsLoss() #BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    best_lwlrap = -np.inf
    early_stop_count = 0

    for epoch in range(args.start_epcoh, args.epochs):
        train_avg, train_loss = train_epoch(args, model, train_loader, criterion, optimizer, scheduler, epoch)
        valid_avg, valid_loss = valid_epoch(args, model, valid_loader, criterion, epoch)

        if args.epoch_scheduler:
            scheduler.step()
        
        content = f"""
                {time.ctime()} \n
                Fold:{args.fold}, Epoch:{epoch}, lr:{optimizer.param_groups[0]['lr']:.7}\n
                Train Loss:{train_loss:0.4f} - LWLRAP:{train_avg['lwlrap']:0.4f}\n
                Valid Loss:{valid_loss:0.4f} - LWLRAP:{valid_avg['lwlrap']:0.4f}\n
        """
        print(content)
        with open(f'{args.save_path}/log_{args.exp_name}.txt', 'a') as appender:
            appender.write(content+'\n')
        
        if valid_avg['lwlrap'] > best_lwlrap:
            print(f"########## >>>>>>>> Model Improved From {best_lwlrap} ----> {valid_avg['lwlrap']}")
            torch.save(model.state_dict(), os.path.join(args.save_path, f'fold-{args.fold}.bin'))
            best_lwlrap = valid_avg['lwlrap']
            early_stop_count = 0
        else:
            early_stop_count += 1
        #torch.save(model.state_dict(), os.path.join(args.save_path, f'fold-{args.fold}_last.bin'))

        if args.early_stop == early_stop_count:
            print("\n $$$ ---? Ohoo.... we reached early stoping count :", early_stop_count)
            break
    
    model.load_state_dict(torch.load(os.path.join(args.save_path, f'fold-{args.fold}.bin'), map_location=args.device))
    model = model.to(args.device)

    target_cols = sub_df.columns[1:].values.tolist()
    test_pred, ids = test_epoch(args, model, test_loader)
    print(np.array(test_pred).shape)

    tmp = pd.DataFrame()
    tmp['recording_id'] = ids
    tmp[target_cols] = test_pred
    test_pred_df = tmp.groupby('recording_id')[target_cols].mean().reset_index()

    test_pred_df.to_csv(os.path.join(args.save_path, f"fold-{args.fold}-submission.csv"), index=False)
    print(os.path.join(args.save_path, f"fold-{args.fold}-submission.csv"))


class args:
    exp_name = "late002"
    pretrain_weights = None
    model_param = {
        'encoder' : 'tf_efficientnet_b2_ns',
        'sample_rate': 48000,
        'window_size' : 512 * 4,
        'hop_size' : 400,
        'mel_bins' : 244,
        'fmin' : 20,
        'fmax' : 48000 // 2,
        'classes_num' : 24
    }
    wave_form_mix_up_ratio = None # 0.9
    period = 10
    seed = CFG.SEED
    start_epcoh = 0 
    epochs = 5 # 30
    lr = 1e-3
    batch_size = 8 # 16
    num_workers = 0
    early_stop = 10
    step_scheduler = True
    epoch_scheduler = False
    num_tta = 10

    device = CFG.DEVICE
    train_csv = CFG.TRAIN_FOLDS_PATH
    train_noisy_csv = CFG.TRAIN_FOLDS_NOISY_PATH
    sub_csv = CFG.SUBMISSION_PATH
    output_dir = "weights"
    train_data_path = CFG.TRAIN_IMG_PATH
    test_data_path = CFG.TEST_IMG_PATH


for use_fold in range(5):
    with trace(f'finetune fold {use_fold}'):
        args.pretrain_weights = f"weights/late001/fold-{use_fold}.bin"
        # print(args.pretrain_weights)
        main(fold=use_fold)

