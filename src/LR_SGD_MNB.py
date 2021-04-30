import numpy as np
import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
import spacy
import re,string,unicodedata
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from time import time
import sys
import warnings
warnings.filterwarnings('ignore')

# helper function to clean text
def token_cleaner(text: str, porterStemmer: PorterStemmer, stopword_list: list) -> str:
    tokens = word_tokenize(text)
    tokens = [porterStemmer.stem(token.lower()) for token in tokens]
    tokens = [token for token in tokens if token not in stopword_list]
    text = ' '.join(tokens)
    pattern=r'[^a-zA-z0-9\s]'
    text=re.sub(pattern, '', text)
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub('\[[^]]*\]', '', text)

    return text

def main(model_to_run: str) -> None: 
    a = time()
    train_set = pd.read_csv('../data/rmp_data/processed/rmp_data_train.csv')
    test_set = pd.read_csv('../data/rmp_data/processed/rmp_data_test.csv')
    porterStemmer = nltk.porter.PorterStemmer()

    # Setting English stopwords
    stopword_list = nltk.corpus.stopwords.words('english')
    # Apply function on review column
    train_set['text'] = train_set['text'].apply(token_cleaner, args=(porterStemmer, stopword_list))
    test_set['text'] = test_set['text'].apply(token_cleaner, args=(porterStemmer, stopword_list))
    # extract text field
    train_set_text = train_set['text']
    test_set_text = test_set['text']
    b = time()
    print("Time cost for reading-in and pre-processing: {0} sec".format(round((b - a), 3)))

    c = time()
    # count vectorizer
    cv = CountVectorizer(min_df=0, max_df=1, binary=False, ngram_range=(1,3))
    # transformed
    cv_train = cv.fit_transform(train_set_text)
    cv_test = cv.transform(test_set_text)

    # tfidf vectorizer
    tv=TfidfVectorizer(min_df=0, max_df=1, use_idf=True, ngram_range=(1,3))
    # transformed
    tv_train = tv.fit_transform(train_set_text)
    tv_test = tv.transform(test_set_text)

    #labeling the sentient data
    lb = LabelBinarizer()
    #transformed sentiment data
    train_sentiments = lb.fit_transform(train_set['label'])
    test_sentiments = lb.fit_transform(test_set['label'])
    d = time()
    print("Time cost for CV, TV, and LB processing: {0} sec".format(round((d - c), 3)))

    ''' Logistic Regression model '''
    def LR() -> None: 
        # training the model
        lr = LogisticRegression(penalty='l2', max_iter=500, C=1, random_state=42)
        # fitting the model
        lr_bow = lr.fit(cv_train, train_sentiments)
        lr_tfidf = lr.fit(tv_train, train_sentiments)

        # predicting the model
        lr_bow_predict = lr.predict(cv_test)
        lr_tfidf_predict = lr.predict(tv_test)

        # accuracy score
        # lr_bow_score = accuracy_score(test_sentiments, lr_bow_predict)
        # print("lr_bow_score :", lr_bow_score)
        # lr_tfidf_score = accuracy_score(test_sentiments, lr_tfidf_predict)
        # print("lr_tfidf_score :", lr_tfidf_score)

        print("=" * 42)
        print("Logistic Regression model")
        # classification report
        lr_bow_report=classification_report(test_sentiments, lr_bow_predict, target_names=['Positive', 'Negative'])
        print(lr_bow_report)
        lr_tfidf_report=classification_report(test_sentiments, lr_tfidf_predict, target_names=['Positive', 'Negative'])
        print(lr_tfidf_report)

        # confusion matrix
        # cm_bow = confusion_matrix(test_sentiments,lr_bow_predict,labels=[1,0])
        # print(cm_bow)
        # cm_tfidf = confusion_matrix(test_sentiments,lr_tfidf_predict,labels=[1,0])
        # print(cm_tfidf)
        return

    ''' Stochastic Gradient Descent '''
    def SGD() -> None: 
        # training the logistic svm
        svm = SGDClassifier(loss='hinge', max_iter=500, random_state=42)
        # fitting the svm
        svm_bow = svm.fit(cv_train, train_sentiments)
        svm_tfidf = svm.fit(tv_train, train_sentiments)

        #Predicting the model
        svm_bow_predict = svm.predict(cv_test)
        svm_tfidf_predict = svm.predict(tv_test)

        # Accuracy score
        # svm_bow_score=accuracy_score(test_sentiments, svm_bow_predict)
        # print("svm_bow_score :", svm_bow_score)
        # svm_tfidf_score=accuracy_score(test_sentiments, svm_tfidf_predict)
        # print("svm_tfidf_score :", svm_tfidf_score)

        print("=" * 42)
        print("Stochastic Gradient Descent")
        #Classification report
        svm_bow_report = classification_report(test_sentiments, svm_bow_predict, target_names=['Positive', 'Negative'])
        print(svm_bow_report)
        svm_tfidf_report = classification_report(test_sentiments, svm_tfidf_predict, target_names=['Positive', 'Negative'])
        print(svm_tfidf_report)

        # confusion matrix
        # cm_bow = confusion_matrix(test_sentiments,svm_bow_predict,labels=[1,0])
        # print(cm_bow)
        # cm_tfidf = confusion_matrix(test_sentiments,svm_tfidf_predict,labels=[1,0])
        # print(cm_tfidf)
        return

    ''' Multinomial Naive Bayes '''
    def MNB() -> None:
        #training the model
        mnb = MultinomialNB()
        #fitting the svm
        mnb_bow = mnb.fit(cv_train, train_sentiments)
        mnb_tfidf = mnb.fit(tv_train, train_sentiments)

        # predicting the model
        mnb_bow_predict = mnb.predict(cv_test)
        print(mnb_bow_predict)
        mnb_tfidf_predict = mnb.predict(tv_test)
        print(mnb_tfidf_predict)

        # Accuracy score
        # svm_bow_score = accuracy_score(test_sentiments, mnb_bow_predict)
        # print("svm_bow_score :", svm_bow_score)
        # svm_tfidf_score = accuracy_score(test_sentiments, mnb_tfidf_predict)
        # print("svm_tfidf_score :", svm_tfidf_score)

        print("=" * 42)
        print("Multinomial Naive Bayes")
        # Classification report
        mnb_bow_report = classification_report(test_sentiments, mnb_bow_predict, target_names=['Positive', 'Negative'])
        print(mnb_bow_report)
        mnb_tfidf_report = classification_report(test_sentiments, mnb_tfidf_predict, target_names=['Positive', 'Negative'])
        print(mnb_tfidf_report)

        # confusion matrix
        # cm_bow = confusion_matrix(test_sentiments, mnb_bow_report, labels=[1,0])
        # print(cm_bow)
        # cm_tfidf = confusion_matrix(test_sentiments, mnb_tfidf_report, labels=[1,0])
        # print(cm_tfidf)
        return

    # conditional execution on command line argument
    if model_to_run == 'lr':
        LR()
    elif model_to_run == 'sgd': 
        SGD()
    elif model_to_run == 'mnb': 
        MNB()
    else: 
        LR()
        SGD()
        MNB()

    return

# always a good practice
if __name__ == '__main__': 
    try: 
        model = sys.argv[1]
        assert(str(model) == 'lr' or str(model) == 'sgd' or str(model) == 'mnb' or str(model) == 'all')
        main(model_to_run=model)
    except (AssertionError, IndexError) as err:  
        print("Usage: \n\tlr for Logistic Regression, sgd for Stochastic Gradient Descent, mnb for Multinomial Naive Bayes, or all for All")