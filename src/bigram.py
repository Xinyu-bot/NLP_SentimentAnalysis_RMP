import pandas as pd
import numpy as np
import os
from time import time
from nltk.tokenize import word_tokenize
from nltk import PorterStemmer
import unigram

# helper function to parse text from bigram dataset and update model accordingly
def process_row(row: tuple, bigram_model: dict, porterStemmer) -> None:
    text, sentiment = row[0], row[1]
    # pos_weight = item_counts[1]
    # negative_weight = item_counts[-1]

    tokens = word_tokenize(text)
    # tokens = [porterStemmer.stem(token) for token in tokens]
    bigrams = [' '.join([tokens[i], tokens[i + 1]]) for i in range(len(tokens)) if i < len(tokens) - 1]
    # print(bigrams)

    for bigram in bigrams: 
        arr = bigram_model.get(bigram, [0, 0])

        # modify the following row 0 or -1 based on the corpus <-- before corpus are standardized! 
        if sentiment == 0: 
            arr[1] += 1
        else: 
            arr[0] += 1

        bigram_model[bigram] = arr    
        
    return

def unigram_backoff(unigram_model: unigram.Lexicon, bigram: list) -> list: 
    ret = []
    # print(bigram)
    
    for token in bigram: 
        try: 
            word_obj = unigram_model._get(token)
            _word, _pos, _sentiment = word_obj.word, word_obj.pos, word_obj.sentiment
            if _sentiment == 'negative': 
                polarity = -1
            elif _sentiment == 'positive': 
                polarity = 1
            else: 
                polarity = 0
        except AttributeError:
            polarity = 0
        
        ret.append(polarity)
    # print(ret)
    _sum = sum(ret)
    if _sum == 0:
        ret = [1, 1]
    elif _sum > 0: 
        ret = [1, 0]
    else: 
        ret = [0, 1]

    # print("unigram backoff result: {0}. ".format(ret))
    return ret

def analyze_bigram(sentence: str, bigram_model: dict, unigram_model: unigram.Lexicon, STOP_WORDS: list, porterStemmer) -> tuple: 
    # print(sentence)
    tokens = word_tokenize(sentence)
    tokens = [token.lower() for token in tokens]
    # tokens = [porterStemmer.stem(token) for token in tokens]
    bigrams = [' '.join([tokens[i], tokens[i + 1]]) for i in range(len(tokens)) if i < len(tokens) - 1]

    pos = 0
    neg = 0
    stop_words = 0
    # loop through each bigram of the current sentence
    for bigram in bigrams: 
        bigram_tokenized = word_tokenize(bigram)
        # remove stop words presence
        if any( (token in STOP_WORDS for token in bigram_tokenized) ): 
            stop_words += 1
            continue

        # retrieve pos and neg occurrence from bigram model
        arr = bigram_model.get(bigram, [-1, -1])
        # if not found, backoff to unigram model
        if arr == [-1, -1]: 
            arr = unigram_backoff(unigram_model, bigram_tokenized)

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
    
    # print(df[df['SentimentText'].isnull()], df.nunique())
    # item_counts = df["Sentiment"].value_counts(normalize=True)
    # print(item_counts)
    # print(df.describe())

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

'''
def main() -> None: 

    dev_set = '../data/bigram/bigram_dev.csv'
    train_set = '../data/bigram/bigram_train.csv'
    test_set = '../data/bigram/bigram_test.csv'
    
    porterStemmer = PorterStemmer()
    bigram_model = process_dev(dev_set, porterStemmer)

    comments = [
        "Adam Meyers is an idiosyncratic individual. Has very dense lectures, but I find them interesting. Obviously not everyone will. Helps to have some prior knowledge of programming. Very knowledgeable and approachable (important for big lectures). Overall great guy with stellar taste in music and fashion.", 
        "Grading was very fair, mostly depend on your understanding of the knowledge instead of how hardworking you seem to be. I barely went to any class bc I was in an 8am section. Attendance was not calculated into the grade. All lectures + modules are accessible on the class website. I understood everything and did all hws+tests well so it was an easy A", 
        "Took his undergrad NLP class. His lectures can be dense (and yes, a little boring), so if you have trouble paying attention to his slides, going to office hours is KEY. He is extremely intelligent in this field, and above all else he wants his students to gain something from the class. The labs are all doable and relevant to the material. 10/10.", 
        "He is probably the best professor out there in NYU. Really lenient on grading criteria and you just have to go to every class and do the problem sets and you will do well. Quizzes get dropped, problem sets get dropped and even one of the two midterms too if you do bad on one. Finals matter the most.", 
        "Very good professor, focuses on learning rather than memorizing, lenient grader. Homeworks are not reaally easy but doable and not that long. All the help is provided in the prompt such as how to approach the problem.", 
        "Having his background in linguistics, he is keen on making python feel useful and accessible to non-stem or non-comp sci majors. Really nice guy. Lectures can get dull sometimes, but he tends to use good examples. Average difficulty homework and exams. The general feel of the class was very laid back. Would take him again!",             
        "Despite the general difficulty of the content, I've learned more in his class than I've learned in other economics classes. As there's no standardized curriculum for this class across the department, he chooses particularly relevant and fundamental topics that have helped me in every other economics class that I've taken. Professor Madsen rocks.", 
        "The best math professor at nyu, no doubt. His lectures are clear and easy to follow along. He doesn't try and trick you on the exams as you are literally tested on what he teaches you from lectures. Homework was a bit more challenging but doable. I took him over the summer, so he taught material faster. If you have the chance to take him, do it!",

        "He teaches so fast that I don't even know what he is saying most of the times. His demonstrations in class are also very unclear because it is on things that we haven't even learned. He literally expects us to know python when this class is for beginners.", 
        "This should be a no prior computer science experience foundation course, but he teaches the course so fast that students without any computer science experience find it difficult to catch with his progress. On the lectures he does programming demonstrations that students even don't have any foundations on.", 
        "Adam is very kind as a person and is easy to approach and talk to. However, as a professor, his teaching style results in extremely tedious and dull lectures that ultimately have turned me away from the subject matter. His lectures are unbelievably monotonous and hard to follow. Homeworks are long and tedious. Exams are a breeze.", 
        "His lectures are painful; he reads dense powerpoints with lots of jargon, and it's unclear what's important and what's filler. The assignments are difficult, but more because there's little to no guidance given. The tests are much easier, so this is a relatively easy class but I'd look elsewhere."
        "Firstly, Madsen is very knowledgable but he tries to prove this. We were given an Intermediate Micro textbook, however we don't follow it and go far beyond the class. Most lectures do not cover the economics topic but dive into multivariable calc with lots of confusing notation. Feel like I can do math but don't actually understand the economics.", 
        "Short and Simple: Do not take if you care about your GPA. Madsen is not a good professor, and his tests are beyond what is learned in class or homework's.", 
        "This guy does not know the first thing about writing exams. He asked students in his Intermediate Microeconomics to build an ALGORITHM on the exam. Who does that? He is a terrible professor and he does not have the slightest clue about the actual caliber of undergraduate students and thinks we're all PhD students back in Stanford with him.", 
        "Professor Madsen is going to ruin your GPA and you are not going to learn. His homework are extremely difficult and his exams are even more exponentially difficult, impossible even. His exams and homework do not resemble the slides he teaches from, which he did not even make. He is the worst professor I've ever taken in my life."
        ]

    for comment in comments: 
        analyze_bigram(comment, bigram_model, unigram_model, porterStemmer)

    return
    
if __name__ == '__main__': 
    main()
''' 