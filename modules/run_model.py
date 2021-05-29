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

def create_quality_models(quality_method = 'gpt_'):
	clean = 'clean_' # Type either '' or 'clean_'
	ns = [10, 50, 100, 500, 2000]
	for n in ns:
		print('\nCreating model for size', n)
		data_type = clean + quality_method + str(n)
		X_all, Y_all = get_data(data_type)
		transformer = OnehotTransformer(ngram_range=(1, 1), min_df=0.0005, max_df=0.5, verbose_vocab=True, max_features=10_000)
		X_all = transformer.fit_transform(X_all)
		model = LogisticRegressionPytorch(input_dim=len(X_all[0]), epochs=200, progress_bar=False)
		batch_size = min(int(len(X_all)*0.1)-1, 4096)
		if batch_size < 10:
			batch_size = 10
		model.train(X_all, Y_all, batch_size=batch_size)
		pickle.dump(model, open(f'../models/{quality_method}model_{n}.obj', 'wb'))
		pickle.dump(transformer, open(f'../models/{quality_method}transformer_{n}.obj', 'wb'))

def plot_deleted_percentage(quality_method = 'gpt_', method='gpt_'):
	ns = [10, 50, 100, 500, 2000][::-1]
	df = pd.DataFrame({'n_base':pd.Series([], dtype='int'),
					   'n_aug':pd.Series([], dtype='int'),
					   'del_p':pd.Series([], dtype='float'),
					   'acc':pd.Series([], dtype='float')})
	for n in ns:
		print('\nRunning size', n)
		clean = 'clean_'
		data_type = clean + method + str(n)
		X_all, Y_all = get_data(data_type)
		
		transformer = pickle.load(open(f'../models/{quality_method}transformer_{n}.obj', 'rb'))
		X = transformer.transform(X_all)
		
		model = pickle.load(open(f'../models/{quality_method}model_{n}.obj', 'rb'))
		with torch.no_grad():
			probs = model.predict_proba(X)
		
		# Doesn't get the "last n"/"original reviews"
		poor_idxs = sorted([((p - l), i) for p, l, i in zip(probs[:,1], Y_all, range(len(probs)-n))], reverse=True)
		start = time()
		ps, scores = [], []
		for del_p in range(100, -1, -5):
			if del_p == 100:
				X = X_all[-n:]
				Y = Y_all[-n:]
			else:
				half_del_size = max(1, int((del_p / 100)*len(poor_idxs))//2)
				del_idxs = set(i for (p, i) in poor_idxs[:half_del_size] + poor_idxs[-half_del_size:])
				print('Deleting', len(del_idxs), 'constituting', len(del_idxs)/len(X_all), 'percent')
				X = [x for i, x in enumerate(X_all) if i not in del_idxs]
				Y = [y for i, y in enumerate(Y_all) if i not in del_idxs]
			
			transformer = OnehotTransformer(ngram_range=(1, 1), min_df=0.0005, max_df=.5, verbose_vocab=True, max_features=4000)
			X = transformer.fit_transform(X)
			model = LogisticRegressionPytorch(input_dim=len(X[0]), epochs=200, progress_bar=False)
			batch_size = min(int(len(X)*0.1)-1, 4096)
			if batch_size < 10:
				batch_size = 10
			
			model.train(X, Y, batch_size=batch_size)
			with torch.no_grad():
				acc = model.score(transformer.transform(Xt), Yt)
			scores.append(acc)
			ps.append(len(X))
			df = df.append({'n_base':n,
						    'n_aug':len(X_all)-n,
						    'del_p':del_p,
						    'acc':acc}, ignore_index=True)
		df.to_csv(f'../results/{quality_method}{method}{n}_delete_poor_idxs.csv')
		plt.plot(ps, scores)
		plt.title('Size: ' + str(n) + ' Time: ' + str(time() - start))
		plt.ylim(0.5, 0.9)
		plt.xticks([n, len(X_all)])
		plt.savefig(f'../imgs/{quality_method}{method}{n}_delete_poor_idxs.png')
		plt.clf()

def main():
	# Uncomment if you want to re-train the quality models
	# create_quality_models(quality_method = 'bert_')
	plot_deleted_percentage(quality_method = 'bert_', method='bert_')

if __name__ == '__main__':
	main()