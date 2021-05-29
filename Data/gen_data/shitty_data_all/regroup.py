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
n_file_paths = [n_base_path + r'\n_{}_for_gpt.txt'.format(size) for size in [10, 50, 100, 500, 2000]]
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

	print(file_name)
	found = False
	for name, n_set in n_sets:
		n = ''.join([c for c in name if c.isnumeric()])
		if len(base_data - n_set) == 0:
			if found:
				raise 'Found multiple matches'
			found = True
			os.system(f'copy {file_name} "../n_{n}_{str(random())}.gpt"')
		
