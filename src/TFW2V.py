import pandas as pd 
import numpy as np
from copy import deepcopy
from string import punctuation
from random import shuffle
import gensim
from gensim.models.word2vec import Word2Vec 
from gensim.models.doc2vec import TaggedDocument
from nltk.tokenize import TweetTokenizer 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize, sent_tokenize
import keras
from keras import Sequential
from keras import *
from keras.layers import Activation, Dense
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

pd.options.mode.chained_assignment = None
tqdm.pandas(desc = "progress-bar")
tokenizer = TweetTokenizer()

# read in the corpus
data1 = pd.read_csv('../data/rmp_data/processed/rmp_data_train.csv')
data2 = pd.read_csv('../data/rmp_data/processed/rmp_data_test.csv')
data = data1.append(data2, ignore_index=True)

data['tokens'] = data['text'].progress_map(word_tokenize)

x_train, x_test, y_train, y_test = train_test_split(np.array(data.tokens), np.array(data.label), test_size=0.2)

def labelizeTweets(tweets, label_type):
    labelized = []
    for i,v in tqdm(enumerate(tweets)):
        label = '%s_%s'%(label_type,i)
        labelized.append(TaggedDocument(v, [label]))
    return labelized

x_train = labelizeTweets(x_train, 'TRAIN')
x_test = labelizeTweets(x_test, 'TEST')

# build word2vec model
n_dim = 200
tweet_w2v = Word2Vec(workers=16, vector_size=200, window=5, min_count=1)
tweet_w2v.build_vocab([x.words for x in tqdm(x_train)])
tweet_w2v.train([x.words for x in tqdm(x_train)], total_examples=tweet_w2v.corpus_count, epochs=tweet_w2v.epochs)

# build TFIDF matrix
print('building tf-idf matrix ...')
vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=10)
matrix = vectorizer.fit_transform([x.words for x in x_train])
tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
print('vocab size :', len(tfidf))

def buildWordVector(tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            vec += tweet_w2v.wv[word].reshape((1, size)) * tfidf[word]
            count += 1.
        except KeyError: # handling the case where the token is not
                         # in the corpus. useful for testing.
            continue
    if count != 0:
        vec /= count
    return vec

from sklearn.preprocessing import scale
n_dim = 200
train_vec_w2v = np.concatenate([buildWordVector(z, n_dim) for z in tqdm(map(lambda x: x.words, x_train))])
train_vec_w2v = scale(train_vec_w2v)

test_vec_w2v = np.concatenate([buildWordVector(z, n_dim) for z in tqdm(map(lambda x: x.words, x_test))])
test_vec_w2v = scale(test_vec_w2v)

# Sequential kicks in
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=200))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_vec_w2v, y_train, epochs=9, batch_size=32, verbose=2)

score = model.evaluate(test_vec_w2v, y_test, batch_size=128, verbose=2)
print(score)