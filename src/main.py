from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk import PorterStemmer
from time import time
import pandas as pd
import numpy as np
from scraper import main as grab
import unigram
import bigram
import vector_similarity
import sys

# self-defined stop words
STOP_WORDS = [] 
# extend stop_words List by nltk stop words
# extension has been removed because it actually lowers precision/recall
# STOP_WORDS += stopwords.words("english")  

def test_vector_similarity() -> None: 
    # define files for the system
    bigram_dev_set = '../data/IMDB_data/Valid.csv'
    bigram_train_set = '../data/IMDB_data/Train.csv'
    bigram_test_set = '../data/IMDB_data/Test.csv'

    df = pd.read_csv(bigram_dev_set, header=0)
    df = df.head(5000)
    
    df_test = pd.read_csv(bigram_test_set, header=0)
    df_test = df_test.head(5000)
    comments = df_test['text'].tolist()
    labels = df_test['label'].tolist()
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

def enhanced_bigram() -> None: 
    # define files for the system
    unigram_file = '../data/subjectivity_clues_hltemnlp05/subjclueslen1-HLTEMNLP05.tff'
    bigram_dev_set = '../data/IMDB_data/Valid.csv'
    bigram_train_set = '../data/IMDB_data/Train.csv'
    bigram_test_set = '../data/IMDB_data/Test.csv'

    a = time()
    unigram_model = unigram.generate_lexicon(unigram_file)
    b = time()
    print("Time cost for unigram model generating: {0} sec".format(round((b - a), 3)))

    c = time()
    porterStemmer = PorterStemmer()
    bigram_model = bigram.process_dev([bigram_dev_set, bigram_train_set], porterStemmer)
    d = time()
    print("Time cost for bigram model generating: {0} sec".format(round((d - c), 3)))

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
        res = bigram.analyze_bigram(text, bigram_model, unigram_model, STOP_WORDS, porterStemmer)
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

def main(): 
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
        "His lectures are painful; he reads dense powerpoints with lots of jargon, and it's unclear what's important and what's filler. The assignments are difficult, but more because there's little to no guidance given. The tests are much easier, so this is a relatively easy class but I'd look elsewhere.", 
        "Firstly, Madsen is very knowledgable but he tries to prove this. We were given an Intermediate Micro textbook, however we don't follow it and go far beyond the class. Most lectures do not cover the economics topic but dive into multivariable calc with lots of confusing notation. Feel like I can do math but don't actually understand the economics.", 
        "Short and Simple: Do not take if you care about your GPA. Madsen is not a good professor, and his tests are beyond what is learned in class or homework's.", 
        "This guy does not know the first thing about writing exams. He asked students in his Intermediate Microeconomics to build an ALGORITHM on the exam. Who does that? He is a terrible professor and he does not have the slightest clue about the actual caliber of undergraduate students and thinks we're all PhD students back in Stanford with him.", 
        "Professor Madsen is going to ruin your GPA and you are not going to learn. His homework are extremely difficult and his exams are even more exponentially difficult, impossible even. His exams and homework do not resemble the slides he teaches from, which he did not even make. He is the worst professor I've ever taken in my life."
        ]

    unigram_file = '../data/subjectivity_clues_hltemnlp05/subjclueslen1-HLTEMNLP05.tff'
    unigram_output = []
    unigram_model = unigram.generate_lexicon(unigram_file)

    bigram_dev_set = '../data/IMDB_data/Valid.csv'
    bigram_train_set = '../data/IMDB_data/Train.csv'
    bigram_test_set = '../data/bigram/bigram_test.csv'
    porterStemmer = PorterStemmer()
    bigram_output = []
    bigram_model = bigram.process_dev([bigram_dev_set, bigram_train_set], porterStemmer)

    for comment in comments: 
        comment_sentiment = unigram.comment_parsing(comment, unigram_model)
        res = unigram.sentiment_analysis(comment_sentiment)
        unigram_output.append(res)

        res = bigram.analyze_bigram(comment, bigram_model, unigram_model, STOP_WORDS, porterStemmer)
        bigram_output.append(res)

    print(unigram_output)
    print(bigram_output)


if __name__ == '__main__': 
    flag = 0
    try: 
        flag = int(sys.argv[1])
    except: 
        pass
    
    if flag == 1:
        enhanced_bigram()
    elif flag == 2:
        test_vector_similarity()