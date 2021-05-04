# Todo: 
# --num_aug=16
# Ntrain	α 		naug
# 500		0.05 	16
# 2,000	0.05 	8
# 5,000	0.1 	4
# More	0.1		4

import os
from shutil import copyfile
from time import sleep

path = '\\'.join( os.getcwd().split('\\')[:-1] ) + '\\Data\\subsets'
subsets = next(os.walk(path))[2]

os.chdir('../../')
os.chdir('eda_nlp')
for file in subsets:
	if file[0] != 'n':
		continue
	extension = '\\eda_' + file
	src = os.getcwd() + '\\data' + extension
	dst = path + extension
	os.system('python code/augment.py --input=' + path + '\\' + file)