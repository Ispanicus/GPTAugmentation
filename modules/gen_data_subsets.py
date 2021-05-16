from get_data import get_data
from random import sample
import re

X, y = get_data()
y = y.where(X.reviewText.str.contains(r'[a-zA-Z]'))
X = X.where(X.reviewText.str.contains(r'[a-zA-Z]'))

ranges = [100, 500, 2000]

neg = X[y == 0]
pos = X[y == 1]

for r in ranges:
	print('\nProducing subset for size:', r)
	text = []
	for label, data in enumerate([neg, pos]):
		idxs = range(len(data))
		idxs = sample(idxs, r//2)
		text += list(data.iloc[idxs].reviewText.str.replace('\n', ' '))
	
	pattern1 = re.compile(r'([^0-9a-zA-Z\s])\1+(?=[a-z0-9A-Z])')
	pattern2 = re.compile(r'([^0-9a-zA-Z\s])\1+')
	pattern3 = re.compile(r'<[^>]>')
	for i, t in enumerate(text):
		t = re.sub(pattern1, r'\1 ', t)
		t = re.sub(pattern2, r'\1', t)
		t = re.sub(pattern3, r'', t)
		text[i] = t
		

	# fix "...blabl -> ... blabl" and reduce "... -> ."
	s_before = len(text)
	text[:] = filter(lambda sen: len(sen) > 35, text)
	print('Removed short sentences:', s_before - len(text))
	
	s_before = len(text)
	new = []
	# Cut long sentences
	for review in text:
		shortened_sentences = 0
		if len(review) > 450:
			extra = review[450:]
			match = re.search(r'[!?.]', extra)
			if match:
				stop = 450 + match.span()[-1]
				review = review[:stop]
				shortened_sentences += 1
			if len(review) > 600:
				continue
		new.append(review)
	print('Shortened long sentences:', shortened_sentences)
	print('Removed difficult long sentences:', s_before - len(new))

	with open(f'../Data/subsets/n_{r}.txt', 'w') as f:
		for i, t in enumerate(text):
			l = int(i > r//2)
			f.write(f'{l}\t{t}\n')

	with open(f'../Data/subsets/n_{r}_for_gpt.txt', 'w') as f:
		for i, t in enumerate(new):
			l = int(i > r//2)
			f.write(f'{l}\t{t}\n')