import pandas as pd
from get_data import get_data
from cleaner import clean_text, even_distribution

def clean_yelp():
	df = pd.read_csv('yelp.txt', sep='\t', encoding='utf8', names=['sentiment', 'reviewText'])
	df = even_distribution(df)
	X = [clean_text(t) for t in df['reviewText']]
	Y = df['sentiment']
	text = '\n'.join([f'{l}\t{t}' for l, t in zip(Y, X)])
	open('../Data/clean_data/other/yelp.txt', 'w').write(text)

def clean_movie():
	df = pd.read_csv('../Data/moviedata.csv', sep=';', encoding='utf8', names=['sentiment', 'reviewText'])
	df.dropna(inplace=True)
	df = even_distribution(df)
	X = [clean_text(t) for t in df['reviewText']]
	Y = df['sentiment']
	text = '\n'.join([f'{l}\t{t}' for l, t in zip(Y, X)])
	open('../Data/clean_data/other/movie.txt', 'w', encoding='utf8').write(text)

def clean_gpt():
	ns = [10, 50, 100, 500, 2000]
	for n in ns:
		limit = n + n*100
		X, Y = get_data("gpt_" + str(n))
		X = X[-limit:]
		Y = Y[-limit:]
		X = [clean_text(t) for t in X]
		text = '\n'.join([f'{l}\t{t}' for l, t in zip(Y, X)])
		open(f'../Data/clean_data/gpt/{n}.txt', 'w', encoding='utf8').write(text)

def clean_bert():
	ns = [10, 50, 100, 500, 2000]
	for n in ns:
		limit = n + n*100
		X, Y = get_data("bert_" + str(n))
		X = X[-limit:]
		Y = Y[-limit:]
		X = [clean_text(t) for t in X]
		text = '\n'.join([f'{l}\t{t}' for l, t in zip(Y, X)])
		open(f'../Data/clean_data/bert/{n}.txt', 'w', encoding='utf8').write(text)

def clean_n():
	ns = [10, 50, 100, 500, 2000]
	for n in ns:
		limit = n + n*100
		X, Y = get_data("n_" + str(n))
		X = [clean_text(t) for t in X]
		text = '\n'.join([f'{l}\t{t}' for l, t in zip(Y, X)])
		open(f'../Data/clean_data/subsets/{n}.txt', 'w', encoding='utf8').write(text)

def clean_eda():
	ns = [10, 50, 100, 500, 2000]
	augs = [64,32,16,8,4]
	for n in ns:
		for aug in augs:
			X, Y = get_data(f"eda_augs_{aug}_n_{n}")
			X = [clean_text(t) for t in X]
			text = '\n'.join([f'{l}\t{t}' for l, t in zip(Y, X)])
			open(f'../Data/clean_data/eda/augs_{aug}_n_{n}.txt', 'w', encoding='utf8').write(text)