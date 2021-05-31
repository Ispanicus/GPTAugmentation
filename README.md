# NLPPhard
NLP Programming is hard is the github repository for the paper "Generative approach to NLP data augmentation using GPT-Neo"


Data Folder
	-clean_data: contains data from all augmentation methods after cleaning

	-eda: raw eda data before cleaning
	-gen_data: raw gpt data before cleaning 
	-subsets: subsets used for generating data augmentation

Imgs
	Folder for result graphs

Results
	Csv files with results


How to clean data:
	-use functions in clean_data.py

How to generate augmented data:
	-run eda.py in modules folder for eda data
	-run bert.py in modules folder for bert data
	-run gpt_samples.py in modules to generate gpt samples
	

How to get results:
	-run run_model.py for BERT and GPT results which will be saved in results and imgs folders
	-run run_model_eda.py for EDA results which will be saved in results and imgs folders



