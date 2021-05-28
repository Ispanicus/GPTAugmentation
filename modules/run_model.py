import os
from train import train, score
from get_data import get_data
from model import LangID, LogisticRegression, ComplementNB, BernoulliNB
from get_gpt_reviews import get_gpt_reviews
import matplotlib.pyplot as plt
from random import shuffle
import numpy as np
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter("ignore", category=ConvergenceWarning)

import sys
import pandas as pd
import seaborn as sns
from tqdm.notebook import trange, tqdm
from model import OnehotTransformer,LogisticRegressionPytorch
import torch
import pickle

Xt, Yt = get_data("dev", cleanText=True)

## PYTORCH - QUALITY CHECK - EXCLUDING 2000
from time import time



def create_complete_models(ns):
	clean = 'clean_' # Type either '' or 'clean_'
	method = "gpt"
	ns = [10, 50, 100, 500, 2000]
	for n in ns:
		print('\nCreating model for size', n)
		data_type = clean + method + f"_{n}"
		X_all, Y_all = get_data(data_type)
		transformer = OnehotTransformer(ngram_range=(1, 1), min_df=0.001, max_df=0.5, verbose_vocab=True)
		transformer.fit(X_all,Y_all)
		X = transformer.transform(X_all)
		model = LogisticRegressionPytorch(input_dim=len(X[0]),epochs=200,progress_bar=False)
		batch_size = min(int(len(X_all)*0.1)-1, 1024)
		if batch_size < 10:
			batch_size = 10
		model.train(X, Y, batch_size=batch_size)
		pickle.dump(model, open(f'model_{n}.obj', 'wb'))

create_complete_models(ns)