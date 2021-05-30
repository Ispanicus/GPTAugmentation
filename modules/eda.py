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
		*outPath, filename =  path.split("/")
		outPath.pop()
		outPath = "/".join(outPath)
		n = int(re.findall(r'[0-9]+', filename)[0])
		if n not in [10, 50]:
			continue
		command = f'python code/augment.py --alpha_sr=0.05 --alpha_rd=0.05 --alpha_ri=0.05 --alpha_rs=0.0 --output={outPath}/eda/augs_{augs}_n_{n}.txt --num_aug={augs} --input={outPath}/clean_data/subsets/{filename[:-4]+ "_for_gpt.txt"}'
		os.system(command)