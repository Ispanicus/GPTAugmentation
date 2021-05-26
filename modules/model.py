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


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression as LR
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import BernoulliNB as BNB, ComplementNB as CNB

class OnehotTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def convert(self, sentence):# [[w1, w2, w3], [w1, w2, w3]]
        output = [0]*len(self.vocab)
        for word in sentence.split():
            word = word.lower()
            if word in self.vocab:
                output[self.vocab[word]] = 1
        return output

    def fit(self, X, y=None):
        vectorizer = TfidfVectorizer()
        vectorizer.fit(X)
        self.vocab = vectorizer.vocabulary_
        return self
    
    def transform(self, X, y=None):
        X_ = [self.convert(row) for row in X]
        return X_


def LogisticRegression(max_iter=-1):
	return Pipeline([
		('onehot', OnehotTransformer()),
		('clf', LR(max_iter=max_iter))
	])
	
def BernoulliNB():
	return Pipeline([
		('onehot', OnehotTransformer()),
		('clf', BNB())
	])

def ComplementNB():
	return Pipeline([
		('onehot', OnehotTransformer()),
		('clf', CNB())
	])