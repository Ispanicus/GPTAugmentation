import json


## Source: https://www.yelp.com/dataset/documentation/main
with open('yelp_academic_dataset_review.json', encoding='utf8') as f, open('yelp.txt', 'w', encoding='utf8') as out:
	lines = 0
	for l in f:
		data = json.loads(l)
		text = data['text'].replace('\n', ' ').replace('\r', ' ')
		stars = int(data['stars'])
		if stars not in (1, 5):
			continue
		sentiment = 1 if stars == 5 else 0
		out.write(f'{sentiment}\t{text}\n')
		lines += 1
		if lines == 500_000:
			break
