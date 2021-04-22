import pandas as pd
import numpy as np
import os
from time import time
from nltk.tokenize import word_tokenize
from nltk import PorterStemmer

# helper function to parse text from bigram dataset and update model accordingly
def process_row(row: tuple, bigram_model: dict, porterStemmer) -> None:
    text, sentiment = row[0], row[1]
    # pos_weight = item_counts[1]
    # negative_weight = item_counts[-1]

    tokens = word_tokenize(text)
    tokens = [porterStemmer.stem(token) for token in tokens]
    bigrams = [' '.join([tokens[i], tokens[i + 1]]) for i in range(len(tokens)) if i < len(tokens) - 1]
    # print(bigrams)

    for bigram in bigrams: 
        arr = bigram_model.get(bigram, [0, 0])

        if sentiment == -1: 
            arr[1] += 1
        else: 
            arr[0] += 1

        bigram_model[bigram] = arr    
        
    return

def analyze_bigram(sentence: str, bigram_model: dict, porterStemmer) -> None: 
    tokens = word_tokenize(sentence)
    tokens = [token.lower() for token in tokens]
    tokens = [porterStemmer.stem(token) for token in tokens]
    bigrams = [' '.join([tokens[i], tokens[i + 1]]) for i in range(len(tokens)) if i < len(tokens) - 1]

    pos = 0
    neg = 0
    for bigram in bigrams: 
        arr = bigram_model.get(bigram, [0, 0])

        _pos, _neg = arr[0], arr[1]
        if _pos + _neg < 5: 
            _pos = 0
            _neg = 0

        pos += _pos
        neg += _neg
    
    count_sum = pos + neg
    try: 
        weight = {
            'positive': round(pos / count_sum, 3), 
            'negative': round(neg / count_sum, 3), 
        }
    except ZeroDivisionError: 
        weight = {
            'positive': 0, 
            'negative': 0, 
        }

    print("This comment has weighed sentiment as: \n\tpositive: {0},\
            \n\tnegative: {1}"
            .format(weight['positive'], weight['negative']))

    # print(bigrams)

    return

# process the developement dataset
def process_dev(filename: str, porterStemmer) -> dict: 
    s = time()
    # read from development dataset
    df = pd.read_csv(filename, header=0)
    # print(df[df['SentimentText'].isnull()], df.nunique())
    # item_counts = df["Sentiment"].value_counts(normalize=True)
    # print(item_counts)

    
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
        
    print(len(bigram_model))

    comments = [
        "Adam Meyers is an idiosyncratic individual. Has very dense lectures, but I find them interesting. Obviously not everyone will. Helps to have some prior knowledge of programming. Very knowledgeable and approachable (important for big lectures). Overall great guy with stellar taste in music and fashion.", 
        "Grading was very fair, mostly depend on your understanding of the knowledge instead of how hardworking you seem to be. I barely went to any class bc I was in an 8am section. Attendance was not calculated into the grade. All lectures + modules are accessible on the class website. I understood everything and did all hws+tests well so it was an easy A", 
        "Took his undergrad NLP class. His lectures can be dense (and yes, a little boring), so if you have trouble paying attention to his slides, going to office hours is KEY. He is extremely intelligent in this field, and above all else he wants his students to gain something from the class. The labs are all doable and relevant to the material. 10/10.", 
        "He is probably the best professor out there in NYU. Really lenient on grading criteria and you just have to go to every class and do the problem sets and you will do well. Quizzes get dropped, problem sets get dropped and even one of the two midterms too if you do bad on one. Finals matter the most.", 
        "Very good professor, focuses on learning rather than memorizing, lenient grader. Homeworks are not reaally easy but doable and not that long. All the help is provided in the prompt such as how to approach the problem.", 
        "Having his background in linguistics, he is keen on making python feel useful and accessible to non-stem or non-comp sci majors. Really nice guy. Lectures can get dull sometimes, but he tends to use good examples. Average difficulty homework and exams. The general feel of the class was very laid back. Would take him again!",             

        "He teaches so fast that I don't even know what he is saying most of the times. His demonstrations in class are also very unclear because it is on things that we haven't even learned. He literally expects us to know python when this class is for beginners.", 
        "This should be a no prior computer science experience foundation course, but he teaches the course so fast that students without any computer science experience find it difficult to catch with his progress. On the lectures he does programming demonstrations that students even don't have any foundations on.", 
        "Adam is very kind as a person and is easy to approach and talk to. However, as a professor, his teaching style results in extremely tedious and dull lectures that ultimately have turned me away from the subject matter. His lectures are unbelievably monotonous and hard to follow. Homeworks are long and tedious. Exams are a breeze.", 
        "His lectures are painful; he reads dense powerpoints with lots of jargon, and it's unclear what's important and what's filler. The assignments are difficult, but more because there's little to no guidance given. The tests are much easier, so this is a relatively easy class but I'd look elsewhere."
        ]
    for comment in comments: 
        analyze_bigram(comment, bigram_model, porterStemmer)

    e = time()
    print("Proccessing development dataset: {0} sec. ".format(round(e - s), 3))
    return bigram_model

def main() -> None: 

    dev_set = '../data/bigram/bigram_dev.csv'
    train_set = '../data/bigram/bigram_train.csv'
    test_set = '../data/bigram/bigram_test.csv'
    
    porterStemmer = PorterStemmer()
    bigram_model = process_dev(dev_set, porterStemmer)

    return
    


if __name__ == '__main__': 
    main()