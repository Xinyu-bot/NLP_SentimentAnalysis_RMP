import os
import pandas as pd
import numpy as np
from time import time
from nltk.tokenize import word_tokenize, sent_tokenize

def process_row(row: tuple, unigram_model: dict) -> None: 
    text, sentiment = row[0], row[1]
    tokens = word_tokenize(text)
    tokens = [token.lower() for token in tokens]

    for unigram in tokens: 
        arr = unigram_model.get(unigram, [0, 0])

        # modify the following row 0 or -1 based on the corpus <-- before corpus are standardized! 
        if sentiment == 0: 
            arr[1] += 1
        else: 
            arr[0] += 1

        unigram_model[unigram] = arr    

    return

# reading in the file
def generate_unigram_model(filenames: list) -> dict: 
    '''
    designed structure: {
        "hello": [positive_occurrence, negative_occurrence], 
        "world": [positive_occurrence, negative_occurrence],\
        ...
    }
    '''
    unigram_model = {}

    a = time()
    df = pd.DataFrame()
    for filename in filenames: 
        temp = pd.read_csv(filename, header=0)
        df = df.append(temp, ignore_index=True)
    b = time()
    print("Time cost for reading files: {0} sec".format(round((b - a), 3)))

    c = time()
    for row in df.itertuples(index=False):
        process_row(row, unigram_model)
    d = time()
    print("Time cost for generating unigram model: {0} sec".format(round((d - c), 3)))

    return unigram_model

# good practice
if __name__ == '__main__': 
    bigram_dev_set = '../data/IMDB_data/Valid.csv'
    bigram_train_set = '../data/IMDB_data/Train.csv'
    bigram_test_set = '../data/IMDB_data/Test.csv'

    unigram_model = generate_unigram_model([bigram_dev_set, bigram_train_set])

    print(len(unigram_model))
    counter = 0
    for k, v in unigram_model.items():
        print(k, v)
        counter += 1
        if counter == 10:
            break