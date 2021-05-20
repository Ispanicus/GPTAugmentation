## To get paths, just import this file and do subset_file_paths.paths to get a list of absolute paths
import os
import re
import pandas as pd

def get_eda_reviews(n):
    if "\\" in os.getcwd():
        path = '\\'.join( os.getcwd().split('\\')[:-1] ) + f'\\Data\\subsets'
    else:
        path = '/'.join(os.getcwd().split('/')[:-1] ) + '/Data/subsets'

    subsets = next(os.walk(path))[2]

    paths = [path.replace('\\', '/') + '/' + file for file in subsets if f"eda_n_{n}" in file]
    data = []

    for path in paths:
        text = open(path, encoding = 'utf-8').read()
        samples = re.split(r'\n', text)
        X = []
        Y = []
        for sample in samples:
            if '\t' not in sample:
                continue
            y,x = sample.split('\t')
            if not x or not y:
                continue
            X.append(x)
            Y.append(y)

        data += [(x, y) for x, y in zip(X, Y)]

    df = pd.DataFrame(data, columns =['reviewText', 'sentiment'])
    return df
