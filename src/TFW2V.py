import pandas as pd 
import numpy as np
from gensim.models.word2vec import Word2Vec 
from gensim.models.doc2vec import TaggedDocument
from nltk.tokenize import word_tokenize 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import scale
import keras
from keras import Sequential
from keras.layers import Dense
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

pd.options.mode.chained_assignment = None
tqdm.pandas(desc = "progress-bar")

# helper function to label text
def labelize(text, label_type):
    labelized = []
    for i,v in tqdm(enumerate(text)):
        label = '{0}_{1}'.format(label_type,i)
        labelized.append(TaggedDocument(v, [label]))

    return labelized

def main() -> None: 
    # read in the corpus
    data1 = pd.read_csv('../data/rmp_data/processed/rmp_data_train.csv')
    data2 = pd.read_csv('../data/rmp_data/processed/rmp_data_test.csv')
    data = data1.append(data2, ignore_index=True)

    # tokenize the text
    data['tokens'] = data['text'].progress_map(word_tokenize)

    # divide the corpus 
    x_train, x_test, y_train, y_test = train_test_split(np.array(data.tokens), np.array(data.label), test_size = 0.2)
    # label the text
    x_train = labelize(x_train, 'TRAIN')
    x_test = labelize(x_test, 'TEST')

    # build word2vec model
    n_dim = 500
    w2v_model = Word2Vec(workers=8, vector_size=n_dim, window=10, min_count=5)
    w2v_model.build_vocab([x.words for x in tqdm(x_train)])
    w2v_model.train([x.words for x in tqdm(x_train)], total_examples=w2v_model.corpus_count, epochs=w2v_model.epochs)

    # build TFIDF matrix
    vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=5)
    matrix = vectorizer.fit_transform([x.words for x in x_train])
    tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))

    # helper function to build word vector based on 
    def buildWordVector(tokens: list, size: int) -> np.array:
        vec = np.zeros(size).reshape((1, size))
        count = 0.
        for word in tokens:
            try:
                vec += w2v_model.wv[word].reshape((1, size)) * tfidf[word]
                count += 1.
            # token is not in corpus
            except KeyError: 
                continue
        if count != 0:
            vec /= count

        return vec

    # build vector
    train_vec_w2v = np.concatenate([buildWordVector(z, n_dim) for z in tqdm(map(lambda x: x.words, x_train))])
    train_vec_w2v = scale(train_vec_w2v)
    test_vec_w2v = np.concatenate([buildWordVector(z, n_dim) for z in tqdm(map(lambda x: x.words, x_test))])
    test_vec_w2v = scale(test_vec_w2v)

    # Sequential kicks in
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=n_dim))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop', 
                  loss='binary_crossentropy', 
                  metrics=[
                      'accuracy', 'MeanSquaredError', 'AUC', keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.TruePositives(), 
                      keras.metrics.TrueNegatives(), keras.metrics.FalsePositives(), keras.metrics.FalseNegatives()
                      ])
    model.fit(train_vec_w2v, y_train, epochs=16, batch_size=32, verbose=0)

    # evaluate the result
    score = model.evaluate(test_vec_w2v, y_test, batch_size=128, verbose=0, workers=8)
    metrics = ['loss', 'accuracy', 'MeanSquaredError', 'AUC', 'Precision', 'Recall', "TruePositives", 'TrueNegatives', 'FalsePositives', 'FalseNegatives']
    print('=' * 42)
    for i, e in enumerate(score): 
        print(metrics[i], e)

    return

# always a good practice
if __name__ == '__main__': 
    main()