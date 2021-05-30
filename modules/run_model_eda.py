from get_data import get_data
import pandas as pd
from model import OnehotTransformer,LogisticRegressionPytorch
import seaborn as sns

ns = [10,50,100,500, 2000][::-1]
augs = [4,8,16,32,64][::-1]


Xt, Yt = get_data(data_type="dev",cleanText=True)

length = len(ns)*len(augs)
data = {"n":[0]*length,
       "augs":[0]*length,
       "score":[0.0]*length,
       "runs":[0]*length}
df = pd.DataFrame(data)

i = 0
for n in ns:
    
    X,Y = get_data(f"clean_n_{n}",early_return=False)

    transformer = OnehotTransformer(ngram_range=(1, 1), min_df=0.0005, max_df=.5, verbose_vocab=True, max_features=9999)
    transformer.fit(X,Y)
    X = transformer.transform(X)

    model = LogisticRegressionPytorch(input_dim=len(X[0]),epochs=200,progress_bar=False)
    batch_size = min(int(len(X)*0.2)-1, 4096)
    if batch_size < 10:
        batch_size = 10
    model.train(X, Y, batch_size=batch_size)

    acc = model.score(transformer.transform(Xt),Yt)
    df.at[i,"n"] = n
    df.at[i,"score"] = acc
    i+=1
    print(f"Finished baseline for n: {n}")
    for aug in augs:
        #print(f"\neda_augs_{aug}_n_{n}")
        X,Y = get_data(data_type=f"clean_eda_augs_{aug}_n_{n}")
        max_features = 99999
        transformer = OnehotTransformer(ngram_range=(1, 1), min_df=0.0005, max_df=.5, verbose_vocab=True, max_features=max_features)
        transformer.fit(X,Y)
        X = transformer.transform(X)

        model = LogisticRegressionPytorch(input_dim=len(X[0]),epochs=200,progress_bar=False)
        batch_size = min(int(len(X)*0.2)-1, 4096)
        if batch_size < 10:
            batch_size = 10
        model.train(X, Y, batch_size=batch_size)

        acc = model.score(transformer.transform(Xt),Yt)
        df.at[i,"n"] = n
        df.at[i, "augs"] = aug
        df.at[i,"score"] = acc
        i+=1
        print(f"Finished for n: {n}, aug: {aug}")

df.to_csv("../results/EDA_results_LR.csv")
g = sns.FacetGrid(df,col="n")
g.map(sns.scatterplot,"augs","score")
g.savefig("../imgs/EDA_results.png")