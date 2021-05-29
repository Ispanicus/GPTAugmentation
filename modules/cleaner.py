from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import subset_file_paths
import re
import pandas as pd
contractions = [("aren't", "are not"),
("can't", "cannot"),
("couldn't", "could not"),
("didn't", "did not"),
("doesn't", "does not"),
("don't", "do not"),
("hadn't", "had not"),
("hasn't", "has not"),
("haven't", "have not"),
("he'd", "he would"),
("he'll", "he will"),
("he's", "he is"),
("here's", "here is"),
("i'd", "i would"),
("i'll", "i will"),
("it's", "it is"),
("i'm", "i am"),
("i've", "i have"),
("isn't", "is not"),
("let's", "let us"),
("mightn't", "might not"),
("mustn't", "must not"),
("shan't", "shall not"),
("she'd", "she would"),
("she'll", "she will"),
("she's", "she is"),
("shouldn't", "should not"),
("that's", "that is"),
("there's", "there is"),
("they'd", "they would"),
("they'll", "they will"),
("they're", "they are"),
("they've", "they have"),
("we'd", "we would"),
("we're", "we are"),
("we've", "we have"),
("weren't", "were not"),
("what'll", "what will"),
("what're", "what are"),
("what's", "what is"),
("what've", "what have"),
("where's", "where is"),
("who'd", "who would"),
("who'll", "who will"),
("who're", "who are"),
("who's", "who is"),
("who've", "who have"),
("won't", "will not"),
("wouldn't", "would not"),
("you'd", "you would"),
("you'll", "you will"),
("you're", "you are"),
("you've", "you have")]




def clean_text(text):
    lemmatizer = WordNetLemmatizer().lemmatize
    text = text.lower()
    pattern1 = re.compile(r'([^0-9a-zA-Z\s])\1+(?=[a-z0-9A-Z])')
    pattern2 = re.compile(r'([^0-9a-zA-Z\s])\1+')
    pattern3 = re.compile(r'<[^>]>')
    text = re.sub(pattern1, r'\1 ', text)
    text = re.sub(pattern2, r'\1', text)
    text = re.sub(pattern3, r'', text)
    for contr, exp in contractions:
        text = text.replace(contr, exp)
    text = text.replace('"', "'")
    text = text.replace("'", '') #remove ' and "
    text = " ".join([lemmatizer(w) for w in word_tokenize(text)])
    return text

def clean_yelp():
	from get_data import even_distribution
	df = pd.read_csv('yelp.txt', sep='\t', encoding='utf8', names=['sentiment', 'reviewText'])
	df = even_distribution(df)
	X = [clean_text(t) for t in df['reviewText']]
	Y = df['sentiment']
	text = '\n'.join([f'{l}\t{t}' for l, t in zip(Y, X)])
	open('../Data/clean_data/other/yelp.txt', 'w').write(text)

def clean_movie():
	from get_data import even_distribution
	df = pd.read_csv('../Data/moviedata.csv', sep=';', encoding='utf8', names=['sentiment', 'reviewText'])
	df.dropna(inplace=True)
	df = even_distribution(df)
	X = [clean_text(t) for t in df['reviewText']]
	Y = df['sentiment']
	text = '\n'.join([f'{l}\t{t}' for l, t in zip(Y, X)])
	open('../Data/clean_data/other/movie.txt', 'w', encoding='utf8').write(text)
	
def clean(path):
    with open(path) as f:
        *prefix, name = path.split('/')
        clean_path = f'{"/".join(prefix)}/clean_{name}'
        with open(clean_path, "w") as outfile:
            for l in f:
                label, text = l.split('\t')
                text = clean_text(text)
                outfile.write(f'{label}\t{text}\n')


def compare_clean_vocab(path):
    *prefix, name = path.split('/')
    clean_path = f'{"/".join(prefix)}/clean_{name}'
    base_vocab, clean_vocab = set(), set()
    for p, vocab in [(path, base_vocab), (clean_path, clean_vocab)]:
        with open(p) as f:
            for l in f:
                label, text = l.split('\t')
                vocab.update(word_tokenize(text))
    print(f"{path}\ncleaned {len(base_vocab):>6} words, removed {len(base_vocab - clean_vocab)}")

if __name__ == "__main__":
    for path in subset_file_paths.paths:
        clean(path)
        compare_clean_vocab(path)