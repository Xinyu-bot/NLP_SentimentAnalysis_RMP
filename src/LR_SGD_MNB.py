import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud,STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize
from bs4 import BeautifulSoup
import spacy
import re,string,unicodedata
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import LancasterStemmer,WordNetLemmatizer
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from textblob import TextBlob
from textblob import Word
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from time import time
import warnings
warnings.filterwarnings('ignore')

a = time()
train_set = pd.read_csv('../data/rmp_data/processed/rmp_data_train.csv')
test_set = pd.read_csv('../data/rmp_data/processed/rmp_data_test.csv')

# Tokenization of text
tokenizer = ToktokTokenizer()
# Setting English stopwords
stopword_list = nltk.corpus.stopwords.words('english')

# Removing the html strips
def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

# Removing the square brackets
def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

# Removing the noisy text
def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text
# Apply function on review column
train_set['text'] = train_set['text'].apply(denoise_text)
test_set['text'] = test_set['text'].apply(denoise_text)

# Define function for removing special characters
def remove_special_characters(text, remove_digits=True):
    pattern=r'[^a-zA-z0-9\s]'
    text=re.sub(pattern,'',text)
    return text

# Apply function on review column
train_set['text'] = train_set['text'].apply(remove_special_characters)
test_set['text'] = test_set['text'].apply(remove_special_characters)

# Stemming the text
def simple_stemmer(text):
    ps = nltk.porter.PorterStemmer()
    text = ' '.join([ps.stem(word) for word in text.split()])
    return text
# Apply function on review column
train_set['text'] = train_set['text'].apply(simple_stemmer)
test_set['text'] = test_set['text'].apply(simple_stemmer)

# set stopwords to english
stop = set(stopwords.words('english'))

# removing the stopwords
def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text

# Apply function on review column
train_set['text'] = train_set['text'].apply(remove_stopwords)
test_set['text'] = test_set['text'].apply(remove_stopwords)

train_set_text = train_set['text']
test_set_text = test_set['text']
b = time()
print("Time cost for reading-in and pre-processing: {0} sec".format(round((b - a), 3)))

c = time()
# Count vectorizer for bag of words
cv = CountVectorizer(min_df=0, max_df=1, binary=False, ngram_range=(1,3))
# transformed
cv_train = cv.fit_transform(train_set_text)
cv_test = cv.transform(test_set_text)


# Tfidf vectorizer
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

''' 
Linear Regression model
'''
# training the model
lr = LogisticRegression(penalty='l2', max_iter=500, C=1, random_state=42)
# Fitting the model
lr_bow = lr.fit(cv_train, train_sentiments)
lr_tfidf = lr.fit(tv_train, train_sentiments)

# Predicting the model
lr_bow_predict = lr.predict(cv_test)
lr_tfidf_predict = lr.predict(tv_test)

# Accuracy score
# lr_bow_score = accuracy_score(test_sentiments, lr_bow_predict)
# print("lr_bow_score :", lr_bow_score)
# lr_tfidf_score = accuracy_score(test_sentiments, lr_tfidf_predict)
# print("lr_tfidf_score :", lr_tfidf_score)

print("=" * 42)
print("Linear Regression model")
# Classification report
lr_bow_report=classification_report(test_sentiments, lr_bow_predict, target_names=['Positive', 'Negative'])
print(lr_bow_report)
lr_tfidf_report=classification_report(test_sentiments, lr_tfidf_predict, target_names=['Positive', 'Negative'])
print(lr_tfidf_report)

'''
Stochastic Gradient Descent
'''
# training the linear svm
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

'''
Multinomial Naive Bayes
'''
#training the model
mnb = MultinomialNB()
#fitting the svm
mnb_bow = mnb.fit(cv_train, train_sentiments)
mnb_tfidf = mnb.fit(tv_train, train_sentiments)

#Predicting the model
svm_bow_predict=svm.predict(cv_test)
svm_tfidf_predict=svm.predict(tv_test)

# Accuracy score
# svm_bow_score = accuracy_score(test_sentiments, svm_bow_predict)
# print("svm_bow_score :", svm_bow_score)
# svm_tfidf_score = accuracy_score(test_sentiments, svm_tfidf_predict)
# print("svm_tfidf_score :", svm_tfidf_score)

print("=" * 42)
print("Multinomial Naive Bayes")
# Classification report
svm_bow_report = classification_report(test_sentiments, svm_bow_predict, target_names=['Positive', 'Negative'])
print(svm_bow_report)
svm_tfidf_report = classification_report(test_sentiments, svm_tfidf_predict, target_names=['Positive', 'Negative'])
print(svm_tfidf_report)

