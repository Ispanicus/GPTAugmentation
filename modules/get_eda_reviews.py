## To get paths, just import this file and do subset_file_paths.paths to get a list of absolute paths
import os
import re
import pandas as pd

def get_eda_reviews(filename):
	filename = filename[4:] + '.txt' #remove eda_
	if "\\" in os.getcwd():
		path = '\\'.join( os.getcwd().split('\\')[:-1] ) + f'\\Data\\eda\\'
		path = path.replace('\\', '/')
	else:
		path = '/'.join(os.getcwd().split('/')[:-1] ) + '/Data/eda/'
	path = path + filename
	lines = open(path, encoding = 'utf-8').readlines()
	fix_type = lambda x: [int(x[0]), x[1].strip()]
	data = [fix_type(l.split('\t')) for l in lines if l]

	df = pd.DataFrame(data, columns =['sentiment', 'reviewText'])
	return df