import re

path = '../Data/gen_data/0.75_negative.txt'


text = open(path).read()
samples = re.split(r'###', text)
X = []
Y = []
for sample in samples:
	x = re.search(r'Amazon review: (.+)', sample.split('Sentiment: ')[0])
	y = re.search(r'Sentiment: ((Negative)|(Positive))', sample)
	if not x or not y:
		continue
	x = x.group(1).strip().lower()
	y = y.group(1)
	X.append(x)
	Y.append(y)

data = [(x, y) for x, y in zip(X, Y)]

print('\n\n'.join(map(str, sorted(data))))
