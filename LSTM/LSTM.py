import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from random import randint 
import sys
#from tqdm import trange

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
print(f"Device used = {device}")

with open("data.txt", encoding="utf-8") as file:
    data = file.readlines()

# Parsing arguments
try:
    if int(sys.argv[1]) == 0:
        print(f"Processing all {len(data)} datapoints")
    else:
        
        data = data[:int(sys.argv[1])]
        print(f"Processing {len(data)} datapoints")
except:
    print(f"Processing all {len(data)} datapoints")
    
try:
    epochs = int(sys.argv[2])
    print(f"Number of epochs = {int(sys.argv[2])}")

except:
    epochs = 10
    print("Number of epochs = 10")


#abtracting from actual words to just numbers
vocab = {"<unknown>":0,"<PAD>": 1}
id = 2

#used for finding the word assoiated with a number
vocab_reverse = {}

#cutoff length for sentences
sentence_length = 20

def normalize(data):
    word2idx = []
    global id, vocab
    for line in data:
        for word in line[:sentence_length]:
            if word not in vocab:
                vocab[word] = id
                id += 1
    for line in data:
        ans = [vocab.get(line[index], vocab["<PAD>"]) for index in range(min(sentence_length, len(line)))]
        for i in range(sentence_length - len(ans)):
            ans.append(vocab["<PAD>"])
        word2idx.append(ans)
    return word2idx

vocab_reverse = {y:x for x,y in vocab.items()}

def fixData(data):
    y = []
    x = []
    for entry in data:
        x_tmp, y_tmp = hideWord(entry)
        if x_tmp:
            x.append(x_tmp)
            y.append(y_tmp)
        #print(vocab)
    return x,y

def hideWord(sentence):
    words = sentence.split()
    if len(words) < 5:
          return False, False
    place = randint(0,len(words)-1)
    hiddenword = words[place]
    words[place] = "<unknown>"
    return (words, hiddenword)

x,y = fixData(data)

x = normalize(x)
for word in y:
    if word not in vocab:
        vocab[word] = id
        id += 1
y = [vocab[label] for label in y]

source = torch.tensor(x)
target = torch.tensor(y)

lstm_dim = 50
embed_dim = 100

class LangID(nn.Module):
    def __init__(self, embed_dim, lstm_dim, vocab_dim):
        super(LangID, self).__init__()
        self.embedding = nn.Embedding(vocab_dim, embed_dim) #id, 100
        self.lstm = nn.LSTM(embed_dim,lstm_dim,batch_first = True, bidirectional = True)
        self.hidden2tag = nn.Linear(2*lstm_dim, vocab_dim)
        self.dropoutlayer = nn.Dropout(0.2)
    
    def forward(self, inputs):

        embeds = self.embedding(inputs)
        #print("embeds",embeds.shape)

        lstm_out, _ = self.lstm(self.dropoutlayer(embeds))
        #print("lstm_out",lstm_out.shape)
      
        tag_space = self.hidden2tag(self.dropoutlayer(lstm_out))[:,-1,:]
        #print("tag_space", tag_space.shape)
        return tag_space
    
    
tmp_feats = source
tmp_labels = target
torch.cuda.empty_cache()
batch_size = 32
num_batches = int(len(tmp_labels)/batch_size)

tmp_feats_batches = tmp_feats[:batch_size*num_batches].view(num_batches,batch_size, sentence_length)
tmp_labels_batches = tmp_labels[:batch_size*num_batches].view(num_batches, batch_size)
tmp_feats_batches = tmp_feats_batches.to(device)
tmp_labels_batches = tmp_labels_batches.to(device)
#creating the model
model = LangID(embed_dim, lstm_dim, id)
model.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

print("Start Training")
for epoch in range(epochs):
    totalloss = 0
    
    for i in range(len(tmp_feats_batches)):
    
        feats_batch = tmp_feats_batches[i]
        labels_batch = tmp_labels_batches[i]
        #print(feats_batch.shape, labels_batch.shape)
        # Here you can call forward/calculate the loss etc.
        model.zero_grad()
        tag_scores = model.forward(feats_batch)

        #print(tag_scores.shape)
        loss = loss_function(tag_scores, labels_batch)
        totalloss += loss.item()
        loss.backward()
        optimizer.step()



print("Completed training")    
torch.save(model,"model")
print("Saved model")   