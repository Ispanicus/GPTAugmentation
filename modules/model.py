import torch.nn as nn
import torch
import torch.optim as optim
from tqdm.notebook import trange, tqdm

class LangID(nn.Module):
    def __init__(self,  vocab_dim,embed_dim=50, lstm_dim=50, dropout=0.2,epochs=3,progress_bar=False):
        super(LangID, self).__init__()
        self.embedding = nn.Embedding(vocab_dim, embed_dim) #id, 100
        self.lstm = nn.LSTM(embed_dim, lstm_dim, batch_first = True, bidirectional = True)
        self.hidden2tag = nn.Linear(2*lstm_dim, 2)
        self.dropoutlayer = nn.Dropout(dropout)
        self.epochs = epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_bar = progress_bar

    def forward(self, inputs):
        inputs = self.embedding(inputs)
        inputs, _ = self.lstm(self.dropoutlayer(inputs))
        inputs = self.hidden2tag(self.dropoutlayer(inputs))[:,-1,:]
        return inputs
		
    def train_(self,X,y,batch_size=64, verbose=False):
        if verbose:
            print("Device:",self.device)
        num_batches = int(len(X)/batch_size)

        X,y = torch.tensor(X),torch.tensor(y)
#         X = X.type(torch.FloatTensor)
        
        source_batches = X[:batch_size*num_batches].view(num_batches,batch_size, len(X[0]))
        target_batches = y[:batch_size*num_batches].view(num_batches, batch_size)


        self.to(self.device)
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=0.002)

        if self.use_bar:
            iterator = trange(self.epochs)
        else:
            iterator = range(self.epochs)
        for _ in iterator:
            for i in range(len(source_batches)):

                feats_batch = source_batches[i].to(self.device)
                labels_batch = target_batches[i].to(self.device)

                self.zero_grad()
                tag_scores = self.forward(feats_batch)
                loss = loss_function(tag_scores, labels_batch)
                loss.backward()
                optimizer.step()
                feats_batch.to("cpu")
                labels_batch.to("cpu")
        return self
    
    def predict(self, X):
        try:
            self.to(self.device)
            X = torch.tensor(X).to(self.device)
        except:
            self.to("cpu")
            X = torch.tensor(X)
        return torch.argmax(self.forward(X), dim=1)

    def predict_proba(self, X):
        try:
            X = torch.tensor(X).to(self.device)
        except:
            X = torch.tensor(X)
        self.eval()
            
        return torch.softmax(self.forward(X), dim=1)

    def score(self, X, y):
        try:
            Xt = torch.tensor(X).to(self.device)
            Yt = torch.tensor(y).to(self.device)
            self.eval()
            preds = torch.argmax(self.forward(Xt), dim=1)
            acc = round((sum(preds == Yt)/len(Yt)).item(), 3)
        except:
            Xt = torch.tensor(X)
            Yt = torch.tensor(y)
            self.eval()
            self.to("cpu")
            preds = torch.argmax(self.forward(Xt), dim=1)
            acc = round((sum(preds == Yt)/len(Yt)).item(), 3)
        
        preds = torch.argmax(self.forward(Xt), dim=1)
        acc = round((sum(preds == Yt)/len(Yt)).item(), 3)
        return acc

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression as LR
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import BernoulliNB as BNB, ComplementNB as CNB

class OnehotTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, ngram_range, min_df, max_df, verbose_vocab, max_features):
        self.ngram_range=ngram_range
        self.min_df=min_df
        self.max_df=max_df
        self.verbose_vocab = verbose_vocab
        self.max_features = max_features

    def convert(self, sentence):# [[w1, w2, w3], [w1, w2, w3]]
        output = [0]*len(self.vocab)
        for word in sentence.split():
            word = word.lower()
            if word in self.vocab:
                output[self.vocab[word]] = 1
        return output

    def fit(self, X, y=None):
        vectorizer = CountVectorizer(ngram_range=self.ngram_range, min_df=self.min_df, max_df=self.max_df, max_features=self.max_features)
        vectorizer.fit(X)
        self.vocab = vectorizer.vocabulary_
        if self.verbose_vocab:
            print('Fitted vocab size:', len(self.vocab))
        return self

    def transform(self, X, y=None):
        X_ = [self.convert(row) for row in X]
        return X_


def LogisticRegression(max_iter=100, ngram_range=(1, 1), min_df=1, max_df=1.0, verbose_vocab=False, max_features=4000):
    return Pipeline([
        ('onehot', OnehotTransformer(ngram_range, min_df, max_df, verbose_vocab, max_features)),
        ('clf', LR(max_iter=max_iter))
    ])

class LogisticRegressionPytorch(torch.nn.Module):
    def __init__(self,input_dim,epochs,progress_bar = False):
        super(LogisticRegressionPytorch, self).__init__()
        self.linear = torch.nn.Linear(input_dim, 2)
        self.epochs = epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_bar = progress_bar


    def forward(self, x):
        outputs = self.linear(x)
        return outputs

    def train(self,X,y,batch_size=64, verbose=False):
        if verbose:
            print("Device:",self.device)
        num_batches = int(len(X)/batch_size)

        X,y = torch.tensor(X), torch.tensor(y)
        X = X.type(torch.FloatTensor)

        source_batches = X[:batch_size*num_batches].view(num_batches,batch_size, len(X[0]))
        target_batches = y[:batch_size*num_batches].view(num_batches, batch_size)


        self.to(self.device)
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=0.002)

        if self.use_bar:
            iterator = trange(self.epochs)
        else:
            iterator = range(self.epochs)
        for _ in iterator:
            for i in range(len(source_batches)):
                feats_batch = source_batches[i].to(self.device)
                labels_batch = target_batches[i].to(self.device)
                self.zero_grad()
                tag_scores = self.forward(feats_batch)
                loss = loss_function(tag_scores, labels_batch)
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X):
        X = torch.tensor(X).type(torch.FloatTensor).to(self.device)
        return torch.argmax(self.forward(X), dim=1)

    def predict_proba(self, X):
        X = torch.tensor(X).type(torch.FloatTensor).to(self.device)
        return torch.softmax(self.forward(X), dim=1)

    def score(self, X, y):
        Xt = torch.tensor(X).type(torch.FloatTensor).to(self.device)
        Yt = torch.tensor(y).to(self.device)
        preds = torch.argmax(self.forward(Xt), dim=1)
        acc = round((sum(preds == Yt)/len(Yt)).item(), 3)
        return acc