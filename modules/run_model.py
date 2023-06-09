import os
from train import train, score
from get_data import get_data
from get_gpt_reviews import get_gpt_reviews
import matplotlib.pyplot as plt
from random import shuffle, sample
import numpy as np
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter("ignore", category=ConvergenceWarning)
from cleaner import clean_text
import sys
import pandas as pd
import seaborn as sns
from tqdm.notebook import trange, tqdm
from model import OnehotTransformer,LogisticRegressionPytorch
import torch
import pickle
import re

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

def plot_deleted_percentage(quality_method='gpt_', method='gpt_', del_p=False, n=False, test=False):
	ns = [10, 50, 100, 500, 2000][::-1] if not n else [n]
	df = pd.DataFrame({'n_base':pd.Series([], dtype='int'),
					   'n_aug':pd.Series([], dtype='int'),
					   'del_p':pd.Series([], dtype='float'),
					   'acc':pd.Series([], dtype='float')})
	for n in ns:
		print('\nRunning size', n)
		clean = 'clean_'
		data_type = clean + method + str(n) + '_for_gpt'
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
		del_percentages = range(100, -1, -5) if not del_p else [del_p]
		for del_p in del_percentages:
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
				if test:
					Xt_ = transformer.transform(Xt)
					return model.score(Xt_, Yt), model.predict(Xt_), Yt
				acc = model.score(transformer.transform(Xt), Yt)
			scores.append(acc)
			ps.append(len(X))
			df = df.append({'n_base':n,
							'n_aug':len(X_all)-n,
							'del_p':del_p,
							'acc':acc}, ignore_index=True)
		plt.plot(ps, scores)
		plt.title('Size: ' + str(n) + ' Time: ' + str(time() - start))
		plt.ylim(0.5, 0.9)
		plt.xticks([n, len(X_all)])
		plt.savefig(f'../imgs/{quality_method}{method}{n}_delete_poor_idxs.png')
		plt.clf()
	df.to_csv(f'../results/{quality_method}{method}_delete_poor_idxs.csv')

def get_sample(r):
	X, Y = get_data()
	X, Y = zip(*[(x.replace('\n', ' '), y) for x, y in zip(X, Y) if re.search(r'[a-zA-Z]', x)])
	neg = [x for x, y in zip(X, Y) if y == 0]
	pos = [x for x, y in zip(X, Y) if y == 1]

	accs = []
	for i in range(100):
		print(i)

		text_neg = sample(list(neg), r//2)
		text_neg = ["0"+x for x in text_neg]
		text_pos = sample(list(pos), r//2)
		text_pos = ["1"+x for x in text_pos]
		text = text_neg+text_pos
		
		Y_all, X_all = zip(*[(int(t[0]), clean_text(t[1:])) for t in text])
		
		transformer = OnehotTransformer(ngram_range=(1, 1), min_df=1, max_df=1., verbose_vocab=True, max_features=4000)
		X_all = transformer.fit_transform(X_all)
		model = LogisticRegressionPytorch(input_dim=len(X_all[0]), epochs=200, progress_bar=False)
		batch_size = min(int(len(X_all)*0.1)-1, 4096)
		if batch_size < 10:
			batch_size = 10

		model.train(X_all, Y_all, batch_size=batch_size)
		with torch.no_grad():
			Xt_ = transformer.transform(Xt)
			accs.append( model.score(Xt_, Yt) )
	print(sum(accs)/len(accs))
	print(accs)

def score():
	global Xt, Yt
	Xt, Yt = get_data("test", cleanText=True)
	ns = [10, 50, 100, 500, 2000]
	results = [['type', 'n', 'score']]
	
	def score_gpt(n, del_p):
		return plot_deleted_percentage(test=True, del_p=del_p, n=n, quality_method = 'gpt_', method='gpt_')
		
	for n in ns:
		score, pred, Yt = score_gpt(n, del_p=100)
		results.append(['base', n, round(score, 3)])
		open(f'../results/test_base_{n}.csv', 'w').write('\n'.join(f'{p}\t{t}' for p, t in zip(pred, Yt)))
	
	def score_bert(n):
		return plot_deleted_percentage(test=True, del_p=0, n=n, quality_method = 'bert_', method='bert_')
	for n in ns:
		score, pred, Yt = score_bert(n)
		results.append(['bert', n, round(score, 3)])
		open(f'../results/test_BERT_{n}.csv', 'w').write('\n'.join(f'{p}\t{t}' for p, t in zip(pred, Yt)))
	
	for n in ns:
		score, pred, Yt = score_gpt(n, del_p=85)
		results.append(['gpt', n, round(score, 3)])
		open(f'../results/test_GPT_{n}.csv', 'w').write('\n'.join(f'{p}\t{t}' for p, t in zip(pred, Yt)))
		
	pd.DataFrame(results).to_csv('../results/test_data.csv', sep='\t', header=False, index=False)


def test_other_datasets():
	global Xt, Yt
	Xt, Yt = get_data("test", cleanText=True)
	for dataset in ('clean_other_yelp', 'clean_other_movie'):
		X, Y = get_data(dataset)

		transformer = OnehotTransformer(ngram_range=(1, 1), min_df=0.0001, max_df=0.8, verbose_vocab=True, max_features=10_000)
		transformer.fit(X,Y)
		X = transformer.transform(X)

		model = LogisticRegressionPytorch(input_dim=len(X[0]),epochs=200,progress_bar=False)
		batch_size = min(int(len(X)*0.1)-1, 4096)
		if batch_size < 10:
			batch_size = 10
		model.train(X, Y, batch_size=batch_size)

		acc = model.score(transformer.transform(Xt),Yt)
		print(acc)

def main():
	pass
	# Uncomment if you want to re-train the quality models
	# create_quality_models(quality_method = 'gpt_')
	
	plot_deleted_percentage(quality_method = 'bert_', method='bert_')
	# score()
	
	
	#global Xt, Yt
	#Xt, Yt = get_data("test", cleanText=True)
	#X_all, Y_all = get_data("clean_n_50")
	#print(X_all)
	#for _ in range(10):
#		#
		#transformer = OnehotTransformer(ngram_range=(1, 1), min_df=1, max_df=1., verbose_vocab=True, max_features=4000)
		#X_all_ = transformer.fit_transform(X_all)
		#model = LogisticRegressionPytorch(input_dim=len(X_all_[0]), epochs=200, progress_bar=False)
		#model.train(X_all_, Y_all, batch_size=1)
		#with torch.no_grad():
		##	Xt_ = transformer.transform(Xt)
		#	print( model.score(Xt_, Yt) )
	

if __name__ == '__main__':
	main()