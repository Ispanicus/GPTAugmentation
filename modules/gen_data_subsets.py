from get_data import get_data
from random import sample

X, y = get_data()
y = y.where(X.reviewText.str.contains(r'[a-zA-Z]'))
X = X.where(X.reviewText.str.contains(r'[a-zA-Z]'))

ranges = [10, 50, 100, 500, 2000]

neg = X[y == 0]
pos = X[y == 1]

for r in ranges:
    text = []
    for label, data in enumerate([neg, pos]):
        idxs = range(len(data))
        idxs = sample(idxs, r//2)
        text += list(data.iloc[idxs].reviewText.str.replace('\n', ' '))

    with open(f'../Data/subsets/n_{r}.txt', 'w') as f:
        for i, t in enumerate(text):
            l = int(i > r//2)
            f.write(f'{l}\t{t}\n')