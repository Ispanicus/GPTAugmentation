import sys
# the mock-0.3.1 dir contains testcase.py, testutils.py & mock.py
from get_data import get_data
import pandas as pd
import seaborn as sns
from tqdm.notebook import trange, tqdm
from model import OnehotTransformer,LangID

ns = [10,50,100,500, 2000]
augs = [4,8,16,32,64]


Xt, Yt = get_data(data_type="dev",cleanText=True)

length = len(ns)*len(augs)
data = {"n":[0]*length,
       "augs":[0]*length,
       "score":[0.0]*length,
       "runs":[0]*length,
       "vocab":[0]*length}
df = pd.DataFrame(data)

i = 0
for n in ns:
    
    X,Y = get_data(f"clean_n_{n}",early_return=False)

    transformer = OnehotTransformer(ngram_range=(1, 1), min_df=0.0005, max_df=.5, verbose_vocab=True, max_features=9999)
    transformer.fit(X,Y)
    X = transformer.transform(X)

    model = LangID(vocab_dim=len(X[0]),epochs=3,progress_bar=False)
    success = False
    batch_sizes = [4096,2048,1024,512,256,128,64]
    z = 0
    while not success:
        try:
            batch_size = min(int(len(X)*0.2)-1, batch_sizes[z])
            if batch_size < 10:
                batch_size = 10
            model.train_(X, Y, batch_size=batch_size)
            success=True
        except:
            z+=1
    acc = model.score(transformer.transform(Xt),Yt)
    df.at[i,"n"] = n
    df.at[i,"vocab"] = len(X[0])
    df.at[i,"score"] = acc
    i+=1
    for aug in augs:
        #print(f"\neda_augs_{aug}_n_{n}")
        if aug == 64 and n == 2000:
            continue
        X,Y = get_data(data_type=f"clean_eda_augs_{aug}_n_{n}")
        max_features = 99999
        transformer = OnehotTransformer(ngram_range=(1, 1), min_df=0.0005, max_df=.5, verbose_vocab=True, max_features=max_features)
        transformer.fit(X,Y)
        X = transformer.transform(X)

        model = model = LangID(vocab_dim=len(X[0]),epochs=3,progress_bar=False)
        batch_sizes = [4096,2048,1024,512,256,128,64]
        z = 0
        success=False
        while not success:
            try:
                batch_size = min(int(len(X)*0.2)-1, batch_sizes[z])
                if batch_size < 10:
                    batch_size = 10
                model.train_(X, Y, batch_size=batch_size)
                success=True
            except:
                z+=1

        acc = model.score(transformer.transform(Xt),Yt)
        df.at[i,"vocab"] = len(X[0])
        df.at[i,"n"] = n
        df.at[i, "augs"] = aug
        df.at[i,"score"] = acc
        i+=1
df.to_csv("EDA_results_LSTM.csv")