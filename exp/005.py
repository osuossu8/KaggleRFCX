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
from src.engine import train_epoch, valid_epoch, test_epoch
from src.losses import PANNsLoss


test_audio_transform = AA.Compose([
    AA.AddGaussianSNR(p=0.5),
    AA.PitchShift(min_semitones=-0.5, max_semitones=0.5, p=0.1),
    AA.Gain(p=0.2)
])


def pseudo_labeling(fold):
    seed_everything(args.seed)

    args.fold = fold
    args.save_path = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(args.save_path, exist_ok=True)
        
    sub_df = pd.read_csv(args.sub_csv).head(100)

    test_dataset = SedDatasetTest(
        df = sub_df,
        period=args.period,
        stride=5,
        audio_transform=test_audio_transform, # None,
        wave_form_mix_up_ratio=None,
        tta=args.num_tta,
        data_path=args.train_data_path,
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
    
    model.load_state_dict(torch.load(os.path.join(args.pretrain_weights), map_location=args.device))
    model = model.to(args.device)

    target_cols = sub_df.columns[1:].values.tolist()
    test_pred, ids = test_epoch(args, model, test_loader)
    print(np.array(test_pred).shape)

    print(np.argmax(test_pred, 1).shape)
 
    tmp = pd.DataFrame()
    tmp['recording_id'] = ids
    tmp['pseudo_species_id'] = np.argmax(test_pred, 1)
    test_pred_df = tmp.groupby('recording_id')['pseudo_species_id'].mean().reset_index()

    sub_df['pseudo_species_id'] = test_pred_df['pseudo_species_id'].copy().astype(np.int16)

    false_species_id_dict = sub_df.groupby('recording_id')['species_id'].apply(list).to_dict()

    def postprocessing(row):
        l = false_species_id_dict[row['recording_id']]
        if row['pseudo_species_id'] in l:
            return row['species_id']
        else:
            return row['pseudo_species_id']

    sub_df['pseudo_species_id'] = sub_df.apply(postprocessing, axis=1)

    # fp の species_id は誤りなので予測結果が fp に一致したものは使用しない
    use_flg = sub_df['species_id']!=sub_df['pseudo_species_id']
    result_df = sub_df[use_flg].dropna()
    result_df['pseudo_species_id'] = result_df['pseudo_species_id'].astype(np.int16)

    print(result_df.shape)
    print(result_df.head())

    result_df.to_csv(os.path.join(args.save_path, f"fold-{args.fold}-pseudo_train_fp.csv"), index=False)
    print(os.path.join(args.save_path, f"fold-{args.fold}-pseudo_train_fp.csv"))


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
    # sub_csv = CFG.SUBMISSION_PATH
    sub_csv = CFG.TRAIN_FOLDS_NOISY_PATH # pseudo-labeling-to-fp
    output_dir = "pseudo_label_fp_output"
    train_data_path = CFG.TRAIN_IMG_PATH
    test_data_path = CFG.TEST_IMG_PATH


for use_fold in range(5):
# if 1:
#     use_fold = 0
    with trace(f'inference {use_fold}'):
        args.pretrain_weights = f"weights_exp003/SED_E0_5F_BASE/fold-{use_fold}.bin"
        print(args.pretrain_weights)
        pseudo_labeling(fold=use_fold)


