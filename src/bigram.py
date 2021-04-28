import pandas as pd
import numpy as np
import os
from time import time
from nltk.tokenize import word_tokenize
from nltk import PorterStemmer
import unigram_lexicon_based

# helper function to parse text from bigram dataset and update model accordingly
def process_row(row: tuple, bigram_model: dict, porterStemmer) -> None:
    text, sentiment = row[0], row[1]
    tokens = word_tokenize(text)
    tokens = [porterStemmer.stem(token.lower()) for token in tokens]

    bigrams = [' '.join([tokens[i], tokens[i + 1]]) for i in range(len(tokens)) if i < len(tokens) - 1]

    for bigram in bigrams: 
        arr = bigram_model.get(bigram, [0, 0])

        # modify the following row 0 or -1 based on the corpus <-- before corpus are standardized! 
        if sentiment == 0: 
            arr[1] += 1
        else: 
            arr[0] += 1

        bigram_model[bigram] = arr    
        
    return

def unigram_backoff(unigram_model: unigram_lexicon_based.Lexicon, bigram: str) -> list: 
    _sum = 0

    # unpack bigram into unigrams
    unigrams = bigram.split(' ')
    for unigram in unigrams: 
        try: 
            word_obj = unigram_model._get(unigram)
            _word, _sentiment = word_obj.word, word_obj.sentiment
            _sentiment = int(_sentiment)
            if _sentiment == 0: 
                polarity = -1
            elif _sentiment == 1: 
                polarity = 1
            else: 
                polarity = 0
        except AttributeError:
            polarity = 0
        
        _sum += polarity

    if _sum == 0:
        ret = [1, 1]
    elif _sum > 0: 
        ret = [1, 0]
    else: 
        ret = [0, 1]

    return ret

def analyze_bigram(sentence: str, bigram_model: dict, unigram_model: unigram_lexicon_based.Lexicon, STOP_WORDS: list, porterStemmer) -> tuple: 
    tokens = word_tokenize(sentence)
    tokens = [porterStemmer.stem(token.lower()) for token in tokens]

    # lazy generate bigrams from sentence
    bigrams = [' '.join([tokens[i], tokens[i + 1]]) for i in range(len(tokens)) if i < len(tokens) - 1]

    pos, neg = 0, 0
    # loop through each bigram of the current sentence
    for bigram in bigrams: 
        # remove stop words presence
        # if any( (token in STOP_WORDS for token in bigram_tokenized) ): 
        #     continue

        # retrieve pos and neg occurrence from bigram model
        arr = bigram_model.get(bigram, [-1, -1])
        # if not found, backoff to unigram model
        if arr == [-1, -1]: 
            arr = unigram_backoff(unigram_model, bigram)

        _pos, _neg = arr[0], arr[1]
        pos += _pos / (_pos + _neg)
        neg += _neg / (_pos + _neg)

    count_sum = pos + neg
    try: 
        # (positive, negative)
        weight = (round(pos / count_sum, 3), round(neg / count_sum, 3))
    except ZeroDivisionError: 
        weight = (0, 0)

    # print("This comment has weighed sentiment as: \n\tpositive: {0}, negative: {1}"
    #        .format(weight[0], weight[1]))

    return tuple(weight)

# process the developement dataset
def process_dev(filenames: list, porterStemmer) -> dict: 
    # read from development dataset
    df = pd.DataFrame()
    for filename in filenames: 
        temp = pd.read_csv(filename, header=0)
        df = df.append(temp, ignore_index=True)
    
    '''
    designed structure: {
        "hello world": [positive_occurrence, negative_occurrence], 
        "world cup": [positive_occurrence, negative_occurrence],\
        ...
    }
    '''
    bigram_model = {}

    # loop through the development set
    for row in df.itertuples(index=False):
        process_row(row, bigram_model, porterStemmer)
  
    return bigram_model
