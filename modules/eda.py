# Todo: 
# --num_aug=16
# Ntrain	α 		naug
# 500		0.05 	16
# 2,000	0.05 	8
# 5,000	0.1 	4
# More	0.1		4
import subset_file_paths
import os

os.chdir('../../')
os.chdir('eda_nlp')
for path in subset_file_paths.paths:
	os.system('python code/augment.py --input=' + path)