print("hello")
import sys
# the mock-0.3.1 dir contains testcase.py, testutils.py & mock.py
import nltk
from get_data import get_data
import pandas as pd
import seaborn as sns
from tqdm.notebook import trange, tqdm
from model import OnehotTransformer,LangID

ns = [10,50,100,500, 2000]
augs = [4,8,16,32,64]

print("Before getting data")
Xt, Yt = get_data(data_type="dev",cleanText=True)

length = len(ns)*len(augs)
data = {"n":[0]*length,
       "augs":[0]*length,
       "score":[0.0]*length,
       "runs":[0]*length,
       "vocab":[0]*length}
df = pd.DataFrame(data)

i = 0
print("before for loop")