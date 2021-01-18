import pandas as pd
import sys
sys.path.append("/root/workspace/KaggleRFCX")

from sklearn.model_selection import StratifiedKFold

from configs import config as CFG
from src.machine_learning_util import trace, seed_everything, to_pickle, unpickle

FOLDS = CFG.FOLDS
SEED = CFG.SEED


with trace('gen train tp folds'):
    train = pd.read_csv(CFG.TRAIN_TP_PATH).sort_values("recording_id")
    ss = pd.read_csv(CFG.SUBMISSION_PATH)

    train_gby = train.groupby("recording_id")[["species_id"]].first().reset_index()
    train_gby = train_gby.sample(frac=1, random_state=SEED).reset_index(drop=True)
    train_gby.loc[:, 'kfold'] = -1

    X = train_gby["recording_id"].values
    y = train_gby["species_id"].values

    kfold = StratifiedKFold(n_splits=FOLDS)
    for fold, (t_idx, v_idx) in enumerate(kfold.split(X, y)):
        train_gby.loc[v_idx, "kfold"] = fold

    train = train.merge(train_gby[['recording_id', 'kfold']], on="recording_id", how="left")
    print(train.kfold.value_counts())
    train.to_csv(CFG.TRAIN_FOLDS_PATH, index=False)


with trace('gen train fp folds'):
    train_fp = pd.read_csv(CFG.TRAIN_FP_PATH)
    print(train_fp.shape)
    train_fp['kfold'] = -1

    for use_fold in range(5):
        val_recording_id = train.query("kfold == @use_fold").recording_id.values
        train_fp.loc[train_fp['recording_id'].isin(val_recording_id), 'kfold'] = use_fold

    train_fp.to_csv(CFG.TRAIN_FOLDS_NOISY_PATH, index=False)

