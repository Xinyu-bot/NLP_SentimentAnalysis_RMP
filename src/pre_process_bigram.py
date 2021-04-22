import pandas as pd
from time import time
from bs4 import BeautifulSoup
import re
from nltk.tokenize import word_tokenize

def clean(x:str, a, b) -> str: 
    output = BeautifulSoup(x, features="html.parser").get_text()
    output = re.sub(a, '', output)
    output = re.sub(b, '', output)

    output = output.lower()

    tokens = word_tokenize(output)

    output = {token.strip('\'"-.') for token in (set(tokens) - \
            set(" ,./><?;':\"][\}{|~`!@#$%^&*)(_+-=")) if len(token) > 1}

    return (' '.join(output)).strip()

def main() -> None:
    s = time()

    df = pd.read_csv("../data/stolen_data/dataset.csv")
    df['Sentiment'] = df['Sentiment'].map(lambda x: -1 if x == 0 else 1)

    df2 = pd.read_csv("../data/stolen_data/training.1600000.processed.noemoticon.csv", names=['Sentiment', 'id', 'date', 'query', 'user', 'SentimentText'])
    df2.drop(columns=['id', 'date', 'query', 'user'], inplace=True)
    df2['Sentiment'] = df2['Sentiment'].map(lambda x: -1 if x == 0 else (0 if x == 2 else 1))
    a = re.compile(r'@[A-Za-z0-9]+')
    b = re.compile(r'https?://[A-Za-z0-9./]+')
    df2['SentimentText'] = df2['SentimentText'].map(lambda x: clean(x, a, b))

    df = df.append(df2, ignore_index=True)

    nan_value = float("NaN")
    df.replace("", nan_value, inplace=True)
    df.dropna(how='any', inplace=True) 

    print(df[df['SentimentText'].isnull()])

    df = df.sample(frac=1).reset_index(drop=True)
    row_count = df['Sentiment'].count()
    
    training_size = row_count * 4 // 5
    development_size = row_count * 1 // 10
    testing_size = row_count - training_size - development_size

    print(training_size, development_size, testing_size, training_size + development_size + testing_size, row_count)

    train_set = df.head(training_size)
    train_set.to_csv('../data/bigram/bigram_train.csv', index=False)

    dev_set = df.head(training_size + development_size)
    dev_set = dev_set.tail(development_size)
    dev_set.to_csv('../data/bigram/bigram_dev.csv', index=False)

    test_set = df.tail(testing_size)
    test_set.to_csv('../data/bigram/bigram_test.csv', index=False)


    e = time()
    print("Pre-processing for dataset cost: {0} seconds".format(round(e - s, 3)))
    return

if __name__ == "__main__": 
    main()