import os
import sys

sys.path.append("/root/workspace/KaggleRFCX")


INPUT_DIR = 'inputs'
FEATURE_DIR = 'features'
OUT_DIR = 'models'
TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
TEST_PATH = os.path.join(INPUT_DIR, "example_test.csv")
LECTURE_PATH = os.path.join(INPUT_DIR, "lectures.csv")
QUESTION_PATH = os.path.join(INPUT_DIR, "questions.csv")
SUBMISSION_PATH = os.path.join(INPUT_DIR, "example_sample_submission.csv")

SEED = 6718


