import numpy as np
import pandas as pd
import json
from get_gpt_reviews import get_gpt_reviews

def get_data(type='train'):
	''' Returns a tuple: (X, target). 
	This is either train, dev, test or hard data 
    fx type = gpt_2000'''
    if "gpt" in type:
        _, n = type.split("_")
        X = get_gpt_reviews(int(n))
        y = X['sentiment']
        X.drop(columns='sentiment', inplace=True)
        return X, y
        
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
	return X, y