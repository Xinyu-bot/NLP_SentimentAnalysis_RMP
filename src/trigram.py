import pandas as pd
import numpy as np
import os
import unigram_lexicon_based
import pickle
import sys
import string
from nltk.corpus import stopwords
from time import time
from nltk.tokenize import word_tokenize
from nltk import PorterStemmer

# Out-of-Model threshold
# if a n-gram sequence occurred lower than this threshold, 
# we consider it as not valid and back-off to a lower n-gram model
OOM_THRESHOLD = 2

# helper function to help remove undesired numbers (with even float type)
def isNumber(string: str) -> bool:
    # try to float the token if no Error happens then it is a number
    try:
        float(string)
        return True
    # if it cannot be floated, it is not a number
    except ValueError:
        return False

# helper function removing undesired tokens and return the cleanned tokens for modulability 
def tokens_cleaner(tokens: list) -> set:
    # evil set subtraction for best performance yet no existing order preserved
    tokens = {token for token in tokens if token not in string.punctuation and not isNumber(token)}
    # (deprecated) return a subset of previous cleaned results by furthur cleaning 
    return tokens

# helper function to parse text from bigram dataset and update model accordingly
def process_row(row: tuple, trigram_model: dict, bigram_model: dict, porterStemmer: PorterStemmer) -> None:
    # unpack the row
    text, sentiment = row[0], row[1]
    tokens = word_tokenize(text)
    tokens = [token.lower() for token in tokens]
    # tokens = tokens_cleaner(tokens)
    tokens = [porterStemmer.stem(token) for token in tokens]

    # loop through tokens    
    index = 0
    while index < len(tokens): 
        # bigram_model update
        if index > 0: 
            bigram = ' '.join((tokens[index - 1], tokens[index]))
            arr_bigram = bigram_model.get(bigram, [0, 0])
            if sentiment == 0: 
                arr_bigram[1] += 1
            else: 
                arr_bigram[0] += 1
            bigram_model[bigram] = arr_bigram
        
        # trigram_model update
        if index > 1: 
            trigram = ' '.join((tokens[index - 2], tokens[index - 1], tokens[index]))
            arr_trigram = trigram_model.get(trigram, [0, 0])
            if sentiment == 0: 
                arr_trigram[1] += 1
            else: 
                arr_trigram[0] += 1
            trigram_model[trigram] = arr_trigram

        # increment index at last
        index += 1

    return

# unigram backoff <-- last defense before neutralizing the individual token
# use lexicon based unigram model, because it provides better result
def unigram_backoff(unigram_model: unigram_lexicon_based.Lexicon, bigram: str) -> tuple: 
    
    temp = []

    # unpack bigram into unigrams
    unigrams = bigram.split(' ')
    for unigram in unigrams: 
        try: 
            word_obj = unigram_model._get(unigram)
            _word, _sentiment = word_obj.word, int(word_obj.sentiment)
            if _sentiment == 0: 
                polarity = -1
            elif _sentiment == 1: 
                polarity = 1
            else: 
                polarity = 0
        except AttributeError:
            polarity = 0
        
        temp.append(polarity)

    # generate returned value
    _sum = sum(temp)
    if _sum == 0:
        ret = (1, 1)
    elif _sum > 0: 
        ret = (1, 0)
    else: 
        ret = (0, 1)
    
    return ret

def bigram_backoff(bigram_model: dict, unigram_model: unigram_lexicon_based.Lexicon, trigram: str) -> tuple: 
    # unpack trigram into two bigrams
    trigram = trigram.split(' ')
    bigrams = [' '.join((trigram[0], trigram[1])), ' '.join((trigram[1], trigram[2]))]

    pos = 0
    neg = 0
    for bigram in bigrams: 
        # retrieve pos and neg occurrence from bigram model
        arr = bigram_model.get(bigram, [-1, -1])
        # if not found or occurrence too few, backoff to unigram model
        if sum(arr) < OOM_THRESHOLD: 
            arr = unigram_backoff(unigram_model, bigram)
        _pos, _neg = arr[0], arr[1]
        pos += _pos / (_pos + _neg)
        neg += _neg / (_pos + _neg)

    # setup returned value
    if pos > neg: 
        ret = (1, 0)
    elif neg > pos:
        ret = (0, 1)
    else: 
        ret = (1, 1)

    return ret

def analyze_trigram(sentence: str, trigram_model: dict, bigram_model: dict, unigram_model: unigram_lexicon_based.Lexicon, STOP_WORDS: list, porterStemmer: PorterStemmer) -> tuple: 
    # clean the input
    tokens = word_tokenize(sentence)
    tokens = [token.lower() for token in tokens]
    # tokens = tokens_cleaner(tokens)
    tokens = [porterStemmer.stem(token) for token in tokens]

    # lazy generate trigrams from sentence
    trigrams = [' '.join((tokens[i], tokens[i + 1], tokens[i + 2])) for i in range(len(tokens)) if i < len(tokens) - 2]

    pos = 0
    neg = 0
    # loop through each trigrams of the current sentence
    for trigram in trigrams: 
        # remove stop words presence directly
        # if any((token in STOP_WORDS for token in trigram)): 
        #    continue

        # retrieve pos and neg occurrence from trigram model
        arr = trigram_model.get(trigram, [-1, -1])
        # if not found or occurrence too few, backoff to unigram model
        if sum(arr) < OOM_THRESHOLD: 
            arr = bigram_backoff(bigram_model, unigram_model, trigram)

        _pos, _neg = arr[0], arr[1]
        pos += _pos / (_pos + _neg)
        neg += _neg / (_pos + _neg)

    count_sum = pos + neg
    try: 
        # in form of (positive, negative)
        weight = (round(pos / count_sum, 3), round(neg / count_sum, 3))
    except ZeroDivisionError: 
        weight = (0, 0)

    return weight

# process the dataset
def train_model(filenames: tuple, porterStemmer: PorterStemmer) -> dict: 
    # read from dataset
    df = pd.DataFrame()
    for filename in filenames: 
        temp = pd.read_csv(filename, header=0)
        df = df.append(temp, ignore_index=True)

    '''
    designed structure: {
        "hello world hi": [positive_occurrence, negative_occurrence], 
        "world cup final": [positive_occurrence, negative_occurrence],\
        ...
    }
    '''
    trigram_model = {}
    bigram_model = {}

    # loop through the set
    for row in df.itertuples(index=False):
        # train model
        process_row(row, trigram_model, bigram_model, porterStemmer)
   
    return trigram_model, bigram_model

def main(regenerate: int, onRMP: int) -> None:
    # initialize basic setups
    unigram_file_extended = '../data/unigram/unigram_lexicon_extended.csv'
    bigram_dev_set = '../data/IMDB_data/Valid.csv'
    bigram_train_set = '../data/IMDB_data/Train.csv'
    bigram_test_set = '../data/IMDB_data/Test.csv'
    RMP_test_set = '../data/rmp_data/rmp_data_small.csv'
    TEST_FILE = RMP_test_set if onRMP else bigram_test_set
    porterStemmer = PorterStemmer()
    STOP_WORDS = []

    a = time()
    unigram_model = unigram_lexicon_based.generate_lexicon(unigram_file_extended)
    b = time()
    print("Time cost for generating lexicon: {0} sec".format(round((b - a), 3)))

    # see the command line instruction -> to regenerate models based on corpus or not
    # maybe model algorithmics have been changed, then we need to regenerate model
    if regenerate: 
        c = time()
        trigram_model, bigram_model = train_model((bigram_dev_set, bigram_train_set), porterStemmer)
        d = time()
        print("Time cost for generating models: {0} sec".format(round((d - c), 3)))

        # dump the models into file, so next time we can read model directly
        _ = time()
        with open('trigram.model', 'wb') as outstream: 
            pickle.dump(trigram_model, outstream)
        with open('bigram.model', 'wb') as outstream: 
            pickle.dump(bigram_model, outstream)
        __ = time()
        print("Time cost for exporting models: {0} sec".format(round((__ - _), 3)))
    # import model from bytefiles to save 10x times from generating model from corpus
    else: 
        _ = time()
        with open('trigram.model', 'rb') as handle:
            trigram_model = pickle.load(handle)
        with open('bigram.model', 'rb') as handle: 
            bigram_model = pickle.load(handle)
        __ = time()
        print("Time cost for importing models: {0} sec".format(round((__ - _), 3)))

    # test our trigram system by the testing set
    e = time()
    # positive counters
    correct_pos, system_pos, label_pos = 0, 0, 0
    # negative counters
    correct_neg, system_neg, label_neg = 0, 0, 0
    # read the testing corpus
    test_df = pd.read_csv(TEST_FILE, header=0)
    # loop through the testing corpus and check the sentiment analysis result and labelled sentiment line by line
    for row in test_df.itertuples(index=False):
        # unpack row
        if not onRMP:
            text, label = row[0], row[1] # for IMDB corpus
        else: 
            text, label = row[1], row[5] # for RMP corpus
        # analyze row based on trigram
        res = analyze_trigram(text, trigram_model, bigram_model, unigram_model, STOP_WORDS, porterStemmer)
        # print("Current line {0}: result {1}, label {2}".format(total, res, label))
        if float(res[0]) > float(res[1]): 
            if int(label) == 1: 
                correct_pos += 1
            system_pos += 1
        elif float(res[0]) < float(res[1]):
            if int(label) == 0: 
                correct_neg += 1
            system_neg += 1
        else: 
            # system cannot determine the polarity, i.e. positive EQUALS TO negative
            pass

        # update label counters
        if int(label) == 1: 
            label_pos += 1
        else: 
            label_neg += 1   
    f = time()
    print("Time cost for system testing: {0} sec".format(round((f - e), 3)))
    print("=" * 42)

    # display performance measures on screen
    # print(correct_recall, total, round((correct_recall / total), 3))
    print("Precision on positive:  {0} / {1} = {2}".format(correct_pos, system_pos, round((correct_pos / system_pos), 3)))
    print("Precision on negative:  {0} / {1} = {2}".format(correct_neg, system_neg, round((correct_neg / system_neg), 3)))
    print("Precision total:        {0} / {1} = {2}".format((correct_pos + correct_neg), (system_pos + system_neg), round(((correct_pos + correct_neg) / (system_pos + system_neg)), 3))) 

    print("Recall on positive:     {0} / {1} = {2}".format(correct_pos, label_pos, round((correct_pos / label_pos), 3)))
    print("Recall on negative:     {0} / {1} = {2}".format(correct_neg, label_neg, round((correct_neg / label_neg), 3)))
    print("Recall total:           {0} / {1} = {2}".format((correct_pos + correct_neg), (label_pos + label_neg), round(((correct_pos + correct_neg) / (label_pos + label_neg)), 3))) 

    print("F-measure on positive:                {0}".format(round((2 / ((1 / (correct_pos/system_pos)) + (1 / (correct_pos/label_pos)))), 3)))
    print("F-measure on negative:                {0}".format(round((2 / ((1 / (correct_neg/system_neg)) + (1 / (correct_neg/label_neg)))), 3)))
    print("F-measure total:                      {0}".format(round((2 / ((1 / ((correct_pos + correct_neg)/(system_pos + system_neg))) + (1 / ((correct_pos + correct_neg)/(system_pos + system_neg))))), 3)))


if __name__ == '__main__': 
    try: 
        main(regenerate=int(sys.argv[1]), onRMP=int(sys.argv[2]))
    except IndexError: 
        print("Usage: \nFirst field: 0 for importing models, 1 for generating models\nSecond field: 0 for IMDB set, 1 for RMP set")
    