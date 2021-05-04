with open("../Data/subsets/n_10.txt") as f:
    outfile = open("../Data/subsets/gpt_10.txt", "w")
    for line in f.readlines():
        t = line.strip().split("\t")
        outfile.write(f'Amazon review: "{t[1]}"\nSentiment: Positive\n###\n')