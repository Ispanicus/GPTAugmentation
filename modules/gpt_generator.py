from random import sample

ranges = [10, 50, 100, 500, 2000]
dic = {"0":"Negative", "1":"Positive"}


for i in ranges:
    with open(f"../Data/subsets/n_{i}.txt") as file:
        cases = file.readlines()
    for sentiment in ["Negative", "Positive"]:
        outfile = open(f"../Data/subsets/gpt_{sentiment}_{i}.txt", "w")
        
        if sentiment == "Negative":
            idx = sample(range(0,i//2), 4)
        else:
            idx = sample(range(i//2,i), 4)
            
        for j in idx:
            _, text = cases[j].strip().split("\t")
            outfile.write(f'Amazon review: "{text}"\nSentiment: {sentiment}\n###\n')
        outfile.write('Amazon review: ')
        outfile.close()