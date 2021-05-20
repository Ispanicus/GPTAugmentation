import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from random import randint 
from tqdm import trange, tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from random import sample
import sys
from model import LangID
import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument("batches", type=int, help=)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
print(f"Device used = {device}")

def normalize(line, vocab, sentence_length):
	#abtracting from actual words to just numbers
	line = line.split() + ["<PAD>"]*sentence_length
	ans = [vocab.get(word, vocab["<PAD>"]) for word in line[:sentence_length]]
	return ans

def train( X, Y, epochs = 20, batch_size=64,embed_dim=100,lstm_dim=100):
	vectorizer = TfidfVectorizer(min_df = 25, max_df = 0.8)
	vectorizer.fit(X)
	vocab = vectorizer.vocabulary_
	vocab = {key:idx+1 for idx,key in enumerate(vocab)}
	vocab["<PAD>"] = 0
	print("Length of vocab:", len(vocab))

	sentence_length = 32
	X = [normalize(line, vocab, sentence_length) for line in X]

	source = torch.tensor(X)
	target = torch.tensor(Y)

	source = source
	target = target
	num_batches = int(len(target)/batch_size)

	source_batches = source[:batch_size*num_batches].view(num_batches,batch_size, sentence_length)
	target_batches = target[:batch_size*num_batches].view(num_batches, batch_size)
	source_batches = source_batches.to(device)
	target_batches = target_batches.to(device)
	#creating the model
	model = LangID(embed_dim, lstm_dim, len(vocab))
	model.to(device)
	loss_function = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=0.002)
	t = trange(epochs, desc='Started Training', leave=True, position=0)

	for epoch in t:
		totalloss = 0
		for i in tqdm(range(len(source_batches)), desc=f'Epoch {epoch+1} progress', leave=False, position=1):
			feats_batch = source_batches[i]
			labels_batch = target_batches[i]

			model.zero_grad()
			tag_scores = model.forward(feats_batch)
			loss = loss_function(tag_scores, labels_batch)
			totalloss += loss.item()
			loss.backward()
			optimizer.step()

		t.set_description(f"Epoch {epoch+1} loss:{totalloss}")
		t.refresh()

	return model, vocab
	
def main():
	from get_data import get_data
	from model import LangID
	X, Y = get_data()
	Xt, Yt = get_data("dev")
	
	sentence_length = 32
	model, vocab = train(X, Y,batch_size=1024, epochs=10)
	Xt = [normalize(line, vocab, sentence_length) for line in Xt]
	Xt, Yt = torch.tensor(Xt).to(device), torch.tensor(Yt).to(device)
	model.eval()
	preds = torch.argmax(model.forward(Xt), dim=1)
	print(f"Accuracy on dev set={(sum(preds == Yt)/len(Yt))}")

if __name__ == '__main__':
	main()
