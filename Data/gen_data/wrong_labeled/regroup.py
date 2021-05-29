import os
from random import random

def get_text_set(file_name):
	text = open(file_name, encoding='utf8').read()
	reviews = text.split('\n')
	base_data = set()
	for review in reviews:
		if not review:
			continue
		review = review.split('\t')[1]
		base_data.add(review)
	return base_data

n_base_path = r'C:\Users\Christoffer\Git\NLPPhard\Data\subsets'
n_file_paths = [n_base_path + r'\n_{}_for_gpt.txt'.format(size) for size in [500]]
n_sets = [(path, get_text_set(path)) for path in n_file_paths]

for file_name in os.listdir():
	if 'n' != file_name[0]:
		continue
	
	text = open(file_name, encoding='utf8').read()
	reviews = text.split('###')[:4]
	base_data = set()
	for review in reviews:
		review = review.split('\nSentiment')[0]
		review = review.strip().lstrip('Amazon review: ')
		base_data.add(review)

	for name, n_set in n_sets:
		if len(base_data - n_set) == 0:
			print()
			os.system('copy ' + file_name + ' ' + 'n_500_' + str(random()) + '.gpt')
			print(file_name)
			print(name)
		
