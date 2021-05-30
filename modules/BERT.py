import nlpaug.augmenter.word as naw
from get_data import get_data
from nlpaug.util import Action
import random

aug = naw.ContextualWordEmbsAug(
    model_path='bert-base-uncased', action="substitute", device="cuda")

ranges = [50]
for r in ranges:
    X,Y = get_data(f"clean_n_{r}_for_gpt")
    with open(f"../Data/clean_data/bert/{r}_{random.random()}.txt", "a+") as f:
        for i in range(10):      
            for x,y in zip(X,Y):
                f.write(f"{y}\t{aug.augment(x)}\n")
        for x,y in zip(X,Y):
            f.write(f"{y}\t{x}\n")