import os
from train import train, score
from get_data import get_data
from model import LangID, LogisticRegression
from get_gpt_reviews import get_gpt_reviews
import matplotlib.pyplot as plt
from random import shuffle
import numpy as np

Xt, Yt = get_data("dev")

for size in [100, 500, 2000]:
    X_all, Y_all = get_data("gpt_" + str(size))
    assert len(get_data("n_" + str(size))) == size # Should be 2000 in the end
    new_len = len(X_all) - size
    print("Augmented 'x' size of original:", new_len/size, "for size", size)
    ps, scores = [], []
    for i in range(0, 101, 10):
        p = i/100
        data_size = int(size + p*new_len) # exclusive to avoid indexing [-1:]
        X, Y = X_all[-data_size:], Y_all[-data_size:]
        model = LogisticRegression(max_iter=1000)
        model.fit(X, Y)
        acc = (model.predict(Xt) == np.array(Yt)).mean()
        scores.append(acc)
        ps.append(p)
    plt.plot(ps, scores)
    plt.savefig(f'{size}_LR.png')
    plt.show()