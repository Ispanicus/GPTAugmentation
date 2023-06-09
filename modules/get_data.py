import numpy as np
import pandas as pd
import json
from get_gpt_reviews import get_gpt_reviews
from get_eda_reviews import get_eda_reviews
from get_bert_reviews import get_bert_reviews
from get_clean_reviews import get_clean_reviews
from cleaner import even_distribution, clean_text
import subset_file_paths
from random import shuffle
import re



def get_data(data_type='train', early_return=False, cleanText=False):
	''' Returns a tuple: (X, target).
	This is either train, dev, test, hard data or movie
	
	for eda data you need to specify the augs size
	e.g. data_type = eda_augs_16_n_100
	
	for gpt you specify the original size
	e.g. data_type = gpt_2000
	
	for clean data, you prepend clean_
	e.g. data_type = clean_eda_augs_16_n_100
	'''
	if 'clean' in data_type:
		X = get_clean_reviews(data_type[len('clean_'):])
	
	elif re.search(r'gpt|eda|bert', data_type):
		if "eda" in data_type:
			X = get_eda_reviews(data_type)
		elif "gpt" in data_type:
			n = data_type.split('_')[1]
			X = get_gpt_reviews(int(n))
		elif "bert" in data_type:
			n = data_type.split('_')[1]
			X = get_bert_reviews(int(n))
		n = data_type.split('_')[-1]
		X = even_distribution(X)
		if 'eda' not in data_type:
			X = X.append(get_data(f'n_{n}', early_return=True), ignore_index=True)
		
	elif "n_" in data_type:
		_, n = data_type.split("_")
		paths = subset_file_paths.paths
		for path in paths:
			*_,filedest = path.split("subsets/")
			if f"n_{n}.txt" == filedest:
				break
		assert f"n_{n}.txt" == filedest, "no such n size exists"    
		data = [line.strip().split("\t") for line in open(path)]
		shuffle(data)
		X = pd.DataFrame(data, columns =['sentiment', 'reviewText'])
		if early_return:
			return X
	elif 'movie' in data_type:
		X = pd.read_csv('../Data/moviedata.csv', sep=';', names=['sentiment', 'reviewText'])
	else:
		paths = {'train' : '../Data/music_reviews_train.json', \
				 'dev'   : '../Data/music_reviews_dev.json', \
				 'test'  : '../Data/music_reviews_test.json', \
				 'hard'  : '../Data/phase_2_masked.json', \
				 }
		path = paths[data_type]

		data = []
		cols = {'verified':0,'reviewTime':1,'reviewerID':2,'asin':3,"reviewText":4,"summary":5,"unixReviewTime":6,"sentiment":7,"id":8}
		for line in open(path):
			review_data = json.loads(line)

			row = [None]*len(cols)
			for key in review_data:
				if key in cols:
					if key == "sentiment":
						row[cols[key]] = 1 if review_data[key] == "positive" else 0
					else:
						row[cols[key]] = str(review_data[key])
			data.append(row)
		X = pd.DataFrame(data, columns=cols)
		# set empty reviews to '' (instead of None)
		X.loc[X['reviewText'].isna(), 'reviewText'] = ''
		X.loc[X['summary'].isna(), 'summary'] = ''
		X = even_distribution(X)
		assert len(X) > 0, "X is empty"
	Y = list(map(int, X['sentiment']))
	X = list(X['reviewText'])
	if cleanText: 
		X = [clean_text(l) for l in X]
	return X, Y
