from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk import PorterStemmer
from time import time
import pandas as pd
import numpy as np
from scraper import main as grab
import unigram_lexicon_based
import unigram_corpus_based
import bigram
import vector_similarity
import sys

# self-defined stop words
STOP_WORDS = [] 
# extend stop_words List by nltk stop words
# extension has been removed because it actually lowers precision/recall
# STOP_WORDS += stopwords.words("english")  

def test_unigram_lexicon_based() -> None: 
    unigram_file = '../data/subjectivity_clues_hltemnlp05/subjclueslen1-HLTEMNLP05.tff'
    bigram_test_set = '../data/IMDB_data/Test.csv'

    unigram_lexicon = unigram_lexicon_based.generate_lexicon(unigram_file) 

    e = time()
    # positive counters
    correct_pos = 0
    system_pos = 0
    label_pos = 0
    # negative counters
    correct_neg = 0
    system_neg = 0
    label_neg = 0
    # read the testing corpus
    test_df = pd.read_csv(bigram_test_set, header=0)
    # loop through the testing corpus and check the sentiment analysis result and labelled sentiment line by line
    for row in test_df.itertuples(index=False):
        text, label = row[0], row[1]
        comment_sentiment = unigram_lexicon_based.comment_parsing(text, unigram_lexicon)
        res = unigram_lexicon_based.sentiment_analysis(comment_sentiment)
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
    
    return

def test_vector_similarity() -> None: 
    # define files for the system
    bigram_dev_set = '../data/IMDB_data/Valid.csv'
    bigram_train_set = '../data/IMDB_data/Train.csv'
    bigram_test_set = '../data/IMDB_data/Test.csv'
    RMP_test_set = '../data/rmp_data/rmp_data_small.csv'

    df = pd.read_csv(bigram_dev_set, header=0)
    df = df.head(5000)
    
    df_test = pd.read_csv(RMP_test_set, header=0)
    # df_test = df_test.head(200)
    comments = df_test['text'].tolist()
    labels = df_test['sentiment'].tolist()
    # print(len(comments), len(labels))

    system_output = vector_similarity.analyze_vector_similarity(df, comments) 

    # positive counters
    correct_pos = 0
    system_pos = 0
    label_pos = 0
    # negative counters
    correct_neg = 0
    system_neg = 0
    label_neg = 0
    # loop through the testing corpus and check the sentiment analysis result and labelled sentiment line by line
    for index, res in enumerate(system_output): 
        if float(res[0]) > float(res[1]): 
            if int(labels[index]) == 1: 
                correct_pos += 1
            system_pos += 1
        elif float(res[0]) < float(res[1]):
            if int(labels[index]) == 0: 
                correct_neg += 1
            system_neg += 1
        else: 
            # system cannot determine the polarity, i.e. positive EQUALS TO negative
            pass

        # update label counters
        if int(labels[index]) == 1: 
            label_pos += 1
        else: 
            label_neg += 1
                
    print("=" * 42)
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

    return


def test_enhanced_bigram() -> None: 
    # define files for the system
    unigram_file = '../data/subjectivity_clues_hltemnlp05/subjclueslen1-HLTEMNLP05.tff'
    unigram_file_extended = '../data/unigram/unigram_lexicon.csv'
    bigram_dev_set = '../data/IMDB_data/Valid.csv'
    bigram_train_set = '../data/IMDB_data/Train.csv'
    bigram_test_set = '../data/IMDB_data/Test.csv'

    # generate unigram model based on lexicon
    a = time()
    unigram_lexicon = unigram_lexicon_based.generate_lexicon(unigram_file_extended)
    b = time()
    print("Time cost for unigram model generating: {0} sec".format(round((b - a), 3)))

    # generate bigram model based on corpus
    c = time()
    porterStemmer = PorterStemmer()
    bigram_model = bigram.process_dev([bigram_dev_set, bigram_train_set], porterStemmer)
    d = time()
    print("Time cost for bigram model generating: {0} sec".format(round((d - c), 3)))

    # test our enhanced bigram system by the testing set
    e = time()
    # positive counters
    correct_pos = 0
    system_pos = 0
    label_pos = 0
    # negative counters
    correct_neg = 0
    system_neg = 0
    label_neg = 0
    # read the testing corpus
    test_df = pd.read_csv(bigram_test_set, header=0)
    # loop through the testing corpus and check the sentiment analysis result and labelled sentiment line by line
    for row in test_df.itertuples(index=False):
        text, label = row[0], row[1]
        res = bigram.analyze_bigram(text, bigram_model, unigram_lexicon, STOP_WORDS, porterStemmer)
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
    
    return

if __name__ == '__main__': 
    flag = 0
    try: 
        flag = int(sys.argv[1])
    except: 
        pass
    
    if flag == 1:
        test_enhanced_bigram()
    elif flag == 2:
        test_vector_similarity()
    # Unfinished yet, DO NOT try
    elif flag == 3: 
        test_unigram_lexicon_based()