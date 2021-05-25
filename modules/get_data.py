import numpy as np
import pandas as pd
import json
from get_gpt_reviews import get_gpt_reviews
from get_eda_reviews import get_eda_reviews
import subset_file_paths
from random import shuffle

def get_data(type='train'):
	''' Returns a tuple: (X, target).
	This is either train, dev, test or hard data
	fx type = gpt_2000

	for eda data you need to specify the augs size
	fx type = eda_augs_16_n_100

	'''
	if "gpt" in type:
		_, n = type.split("_")
		X = get_gpt_reviews(int(n))
		X = X.append(get_data(f'n_{n}'), ignore_index=True)

	elif "eda" in type:
		*trash, augs,trash1,n = type.split("_")
		X = get_eda_reviews(int(n), int(augs))
		X = X.append(get_data(f'n_{n}'), ignore_index=True)

	elif "n_" in type:
		'''WARNING: THIS ONE RETURNS STUFF'''
		_, n = type.split("_")
		paths = subset_file_paths.paths
		for path in paths:
			*_,filedest = path.split("subsets/")
			if f"n_{n}.txt" == filedest:
				break
		data = [line.strip().split("\t") for line in open(path)]
		shuffle(data)
		X = pd.DataFrame(data, columns =['sentiment', 'reviewText'])
		return X


	else:
		paths = {'train' : '../Data/music_reviews_train.json', \
				 'dev'   : '../Data/music_reviews_dev.json', \
				 'test'  : '../Data/music_reviews_test_masked.json', \
				 'hard'  : '../Data/phase_2_masked.json', \
				 }
		path = paths[type]

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
	y = X['sentiment']
	X.drop(columns='sentiment', inplace=True)
	return list(X["reviewText"]), [int(y_) for y_ in y]
