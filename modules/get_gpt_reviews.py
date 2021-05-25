## To get paths, just import this file and do subset_file_paths.paths to get a list of absolute paths
import os
import re
import pandas as pd

def get_gpt_reviews(n):
    if "\\" in os.getcwd():
        path = '\\'.join( os.getcwd().split('\\')[:-1] ) + f'\\Data\\gen_data'
    else:
        path = '/'.join(os.getcwd().split('/')[:-1] ) + '/Data/gen_data'
        
    subsets = next(os.walk(path))[2]

    paths = [path.replace('\\', '/') + '/' + file for file in subsets if f"n_{n}_" in file]
    data = []
    
    for path in paths:
        text = open(path, encoding = 'utf-8').read()
        samples = re.split(r'###', text)[4:]
        X = []
        Y = []
        for sample in samples:
            x = re.search(r'Amazon review:(.+)', sample.split('Sentiment:')[0])
            y = re.search(r'Sentiment: ((Negative)|(Positive))', sample)
            if not x or not y:
                continue
            x = x.group(1).strip().lower()
            y = 0 if y.group(1) == "Negative" else 1
            X.append(x)
            Y.append(y)

        data += [(x, y) for x, y in zip(X, Y)]

    df = pd.DataFrame(data, columns =['reviewText', 'sentiment'])

    return df
