import numpy as np
import pandas as pd
import json
from get_gpt_reviews import get_gpt_reviews
from get_eda_reviews import get_eda_reviews
from get_clean_reviews import get_clean_reviews
import subset_file_paths
from cleaner import clean_text
from random import shuffle
import re

def get_data(data_type='train', early_return=False, cleanText = False):
    ''' Returns a tuple: (X, target).
    This is either train, dev, test or hard data
    e.g. data_type = gpt_2000

    for eda data you need to specify the augs size
    e.g. data_type = eda_augs_16_n_100

    for clean data, you prepend clean_
    e.g. data_type = clean_eda_augs_16_n_100
    '''
    def even_distribution(X):
        positive = sum(X['sentiment'] == 1)
        L = min(positive, len(X) - positive)
        X = X[X.sentiment == 0][:L].append(X[X.sentiment == 1][:L]) # Ensure even distribution
        X = X.sample(frac = 1) # Shuffle
        assert len(X) != 0
        return X

    if 'clean' in data_type:
        X = get_clean_reviews(data_type.lstrip('clean_'))

    elif re.search(r'gpt|eda', data_type):
        if "eda" in data_type:
            X = get_eda_reviews(data_type)
        else:
            _,n = data_type.split("_")
            X = get_gpt_reviews(int(n))

        n = data_type.split('_')[-1]
        X = even_distribution(X)
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

    else:
        paths = {'train' : '../Data/music_reviews_train.json', \
                 'dev'   : '../Data/music_reviews_dev.json', \
                 'test'  : '../Data/music_reviews_test_masked.json', \
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
    Y = X['sentiment']
    X.drop(columns='sentiment', inplace=True)

    if cleanText: 
        X = [clean_text(ele) for ele in X["reviewText"]]
    else: 
        X = list(X["reviewText"])

    return X, [int(y) for y in Y]
