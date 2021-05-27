import os
import re
import pandas as pd

def get_clean_reviews(n, data_type="n_2000"):
	folder, *filename = data_type.split('_')
	filename = '_'.join(filename) + '.txt'
	
	if folder == 'n':
		filename = 'n_' + filename
		folder = 'subsets'
	
	if "\\" in os.getcwd():
		path = '\\'.join( os.getcwd().split('\\')[:-1] ) + f'\\Data\\clean_data\\{folder}\\'
		path = path.replace('\\', '/')
	else:
		path = '/'.join(os.getcwd().split('/')[:-1] ) + f'/Data/clean_data/{folder}/'
	path += filename
	
	lines = open(path, encoding = 'utf-8').readlines()
	fix_type = lambda x: [int(x[0]), x[1].strip()]
	data = [fix_type(l.split('\t')) for l in lines if l]

	df = pd.DataFrame(data, columns =['sentiment', 'reviewText'])
	return df