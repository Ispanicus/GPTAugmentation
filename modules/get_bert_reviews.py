import os
import re
import pandas as pd

def get_bert_reviews(n):
    if "\\" in os.getcwd():
        path = '\\'.join( os.getcwd().split('\\')[:-1] ) + f'\\Data\\gen_data\\BERT'
    else:
        path = '/'.join(os.getcwd().split('/')[:-1] ) + '/Data/gen_data/BERT'
        
    subsets = next(os.walk(path))[2]
    
    paths = [path.replace('\\', '/') + '/' + file for file in subsets if f"{n}.txt" in file]
    data= []
    assert len(paths) > 0, "no BERT data with this n size"
    for path in paths:
        text = open(path, encoding = 'utf-8').read()
        samples = re.split(r'\n', text)
        X = []
        Y = []
        for sample in samples[:-(n)]:
            y_tmp,x_tmp =  sample.split("\t")
            if not x_tmp or not y_tmp:
                continue
            Y.append(int(y_tmp))
            X.append(x_tmp)
            
        data += [(x, y) for x, y in zip(X, Y)]

    df = pd.DataFrame(data, columns =['reviewText', 'sentiment'])
    return df