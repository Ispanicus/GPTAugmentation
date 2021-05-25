import torch.nn as nn

class LangID(nn.Module):
    def __init__(self, embed_dim, lstm_dim, vocab_dim, dropout):
        super(LangID, self).__init__()
        self.embedding = nn.Embedding(vocab_dim, embed_dim) #id, 100
        self.lstm = nn.LSTM(embed_dim, lstm_dim, batch_first = True, bidirectional = True)
        self.hidden2tag = nn.Linear(2*lstm_dim, 2)
        self.dropoutlayer = nn.Dropout(dropout)
    
    def forward(self, inputs):
        embeds = self.embedding(inputs)
        lstm_out, _ = self.lstm(self.dropoutlayer(embeds))
        tag_space = self.hidden2tag(self.dropoutlayer(lstm_out))[:,-1,:]
        return tag_space
