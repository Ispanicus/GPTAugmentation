from random import sample
import pandas as pd
from transformers import pipeline, GPTNeoForCausalLM, AutoTokenizer
import torch
from random import random

n_type = 2000 # Choose subset size
sizes = [30]*20
for index, size in enumerate(sizes):
    df = pd.read_csv(f"../subsets/n_{n_type}_for_gpt.txt", sep="\t", names=["sentiment","text"])
    df["sentiment"] = pd.to_numeric(df["sentiment"])
    negmask = df["sentiment"] == 0
    posmask = df["sentiment"] == 1
    neg = df[negmask]["text"]
    pos = df[posmask]["text"]
    model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B").to("cuda")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
    
    with open(f"/home/alai/data/n_{n_type}_size_{size}_{random.random()}.gpt", "w") as outfile:
        for _ in range(size):
            negative = sample(list(neg), 2)
            positive = sample(list(pos), 2)
        
            text = []
            for t in positive:
                text.append(f'Amazon review: {t}\nSentiment: Positive\n###\n')
            for t in negative:
                text.append(f'Amazon review: {t}\nSentiment: Negative\n###\n')
            text = "".join(text) + "Amazon review:"
        
            ids = tokenizer(text, return_tensors="pt").input_ids.to("cuda")
            max_length = 1000 + ids.shape[1] # add the length of the prompt tokens to match with the mesh-tf generation
        
            gen_tokens = model.generate(
                ids,
                min_length=max_length-50,
                do_sample=True,
                max_length=max_length,
                temperature=0.75,
                no_repeat_ngram_size=15,
                use_cache=True
            )
            outfile.write(tokenizer.batch_decode(gen_tokens)[0])
            outfile.write("\n\nNEW SAMPLE\n\n")