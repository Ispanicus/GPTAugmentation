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