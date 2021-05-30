import sys
import os
# the mock-0.3.1 dir contains testcase.py, testutils.py & mock.py
from get_data import get_data
from train import train,score
import pandas as pd
import seaborn as sns

X,Y = get_data(f"clean_n_10",early_return=False)
os.chdir('../../../../')
sys.path.append('/usr/local/bin')
print(os.getcwd())
from textattack.augmentation import EmbeddingAugmenter
augmenter = augmentation.EmbeddingAugmenter()
augmented_data = augmentation.augmenter.augment_many(X)

print(augmented_data)