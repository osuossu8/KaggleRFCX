import numpy as np   
import pandas as pd
import seaborn as sns
import os
import sys
import gc
import logging
import random
import warnings
warnings.simplefilter('ignore', FutureWarning)

from tqdm import tqdm
tqdm.pandas()

sys.path.append("/root/workspace/KaggleRFCX")
import riiideducation
from configs import config as CFG
from src.machine_learning_util import timer, seed_everything, to_pickle, unpickle


seed_everything(CFG.SEED)

with timer('load raw train'):
    train = pd.read_csv(CFG.TRAIN_PATH,
                   dtype={'row_id': 'int64',
                          'timestamp': 'int64',
                          'user_id': 'int32',
                          'content_id': 'int16',
                          'content_type_id': 'int8',
                          'task_container_id': 'int16',
                          'user_answer': 'int8',
                          'answered_correctly':'int8',
                          'prior_question_elapsed_time': 'float32',
                          'prior_question_had_explanation': 'boolean'}
                   )


print(train.shape)
print(train.head())

with timer('to pickle raw train'):
    to_pickle(os.path.join(CFG.INPUT_DIR, 'train_raw.pkl'), train)

with timer('load pickled train'):
    train = unpickle(os.path.join(CFG.INPUT_DIR, 'train_raw.pkl'))

print(train.shape)
print(train.head())
print('finished')



