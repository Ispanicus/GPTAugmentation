# Todo: 
# --num_aug=16
# Ntrain	α 		naug
# 500		0.05 	16
# 2,000	0.05 	8
# 5,000	0.1 	4
# More	0.1		4
import subset_file_paths
import os
import re
os.chdir('../../')
os.chdir('eda_nlp')
for augs in [64,32,16,8,4]:
	for path in subset_file_paths.paths:
		*localPath, filename =  path.split("/")
		#localPath.insert(-1, "clean_data")
		localPath = "/".join(localPath)
		n = int(re.findall(r'[0-9]+', filename)[0])
		os.system('python code/augment.py --output={3}/eda/augs_{0}_n_{1} --num_aug={0} --input={2}'.format(augs,n,path, localPath))