import torch.nn as nn
import torch

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
    def __init__(self, ngram_range, min_df, max_df, verbose_vocab):
        self.ngram_range=ngram_range
        self.min_df=min_df
        self.max_df=max_df
        self.verbose_vocab = verbose_vocab

    def fit(self, X, y=None):
        vectorizer = TfidfVectorizer(ngram_range=self.ngram_range, min_df=self.min_df, max_df=self.max_df)
        vectorizer.fit(X)
        self.vocab = vectorizer.vocabulary_
        if self.verbose_vocab:
            print('Fitted vocab size:', len(self.vocab))
        return self

    def transform(self, X, y=None):
        X_ = [[0]*len(self.vocab) for _ in X]
        for i, sent in enumerate(X):
            for word in sent.lower().split():
                if word in self.vocab:
                    X_[i][self.vocab[word]] = 1
        return X_


def LogisticRegression(max_iter=100, ngram_range=(1, 1), min_df=1, max_df=1.0, verbose_vocab=False):
	return Pipeline([
		('onehot', OnehotTransformer(ngram_range, min_df, max_df, verbose_vocab)),
		('clf', LR(max_iter=max_iter))
	])
	
def BernoulliNB(ngram_range=(1, 1), min_df=1, max_df=1.0, verbose_vocab=False):
	return Pipeline([
		('onehot', OnehotTransformer(ngram_range, min_df, max_df, verbose_vocab)),
		('clf', BNB())
	])

def ComplementNB(ngram_range=(1, 1), min_df=1, max_df=1.0, verbose_vocab=False):
	return Pipeline([
		('onehot', OnehotTransformer(ngram_range, min_df, max_df, verbose_vocab)),
		('clf', CNB())
	])

class LogisticRegressionPytorch(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionPytorch, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs