import os
import sys

sys.path.append("/root/workspace/KaggleRFCX")

INPUT_DIR = 'inputs'
OUT_DIR = 'models'
INTERMID_DIR = 'intermids'
TRAIN_IMG_PATH = os.path.join(INPUT_DIR, "train")
TEST_IMG_PATH = os.path.join(INPUT_DIR, "train")
TRAIN_FP_PATH = os.path.join(INPUT_DIR, "train_fp.csv")
TRAIN_TP_PATH = os.path.join(INPUT_DIR, "train_tp.csv")
SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

TRAIN_FOLDS_PATH = os.path.join(INTERMID_DIR, "train_folds.csv")
TRAIN_FOLDS_NOISY_PATH = os.path.join(INTERMID_DIR, "train_folds_noisy.csv")

SEED = 6718
FOLDS = 5

