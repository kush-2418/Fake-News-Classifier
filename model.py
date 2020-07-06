#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 13:22:46 2020

@author: kush
"""


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib qt5
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud,STOPWORDS
from bs4 import BeautifulSoup
import re,string,unicodedata
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from string import punctuation
import itertools

nltk.download('punkt')
nltk.download('wordnet')
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer
import re,re


df = pd.read_csv('/Users/kush/Downloads/Fake News Classifier/train.csv')
df.head()

df = df.drop(['id','title','author'],axis=1)
df = df.dropna()

stop_words = set(stopwords.words('english'))
punctuation = list(string.punctuation)
stop_words.update(punctuation)# Data Cleaning
ps = PorterStemmer()
lemma = WordNetLemmatizer()

def clean_text(text):
    soup = BeautifulSoup(text,'html.parser').get_text()
    text = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", '', text)
    text = re.sub('[^A-Z a-z 0-9-]+',' ',text)
    text = re.sub('[.!-]+',' ',text)
    text = re.sub(r'http\S+','',text)
    text = re.sub('[\d]','',text)
    text = text.lower()

    text = [t for t in text.split() if t not in stop_words]    
    lem_words = [lemma.lemmatize(i) for i in text]

    text = ' '.join(lem_words)
    return text



df['text'] = df['text'].apply(lambda x: clean_text(x))
df['text']


df['len'] = df['text'].apply(lambda s : len(s))
df['len'].plot.hist(bins=100)
df.len.quantile(0.75)


# Wordcloud
plt.figure(figsize = (8,8)) # Text that is not fake
wc = WordCloud(max_words = 200 , width = 2000 , height = 1600).generate(" ".join(df[df.label == 0].text))
plt.title('Wordcloud for Not Fake News')
plt.imshow(wc , interpolation = 'bilinear')


plt.figure(figsize = (8,8)) # Text that is fake
wc = WordCloud(max_words = 200 , width = 2000 , height = 1600).generate(" ".join(df[df.label == 1].text))
plt.title('Wordcloud for Fake News')
plt.imshow(wc , interpolation = 'bilinear')



#Number of characters in texts


fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))
text_len=df[df['label']==1]['text'].str.len()
ax1.hist(text_len,color='red')
ax1.set_title('Fake text')
text_len=df[df['label']==0]['text'].str.len()
ax2.hist(text_len,color='green')
ax2.set_title('Not Fake text')
fig.suptitle('Characters in texts')
plt.show()

# Number of words in each text

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))
text_len=df[df['label']==1]['text'].str.split().map(lambda x: len(x))
ax1.hist(text_len,color='red')
ax1.set_title('Fake text')
text_len=df[df['label']==0]['text'].str.split().map(lambda x: len(x))
ax2.hist(text_len,color='green')
ax2.set_title('Not Fake text')
fig.suptitle('Words in texts')
plt.show()


# Average word length in a text

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(20,10))
word=df[df['label']==1]['text'].str.split().apply(lambda x : [len(i) for i in x])
sns.distplot(word.map(lambda x: np.mean(x)),ax=ax1,color='red')
ax1.set_title('Fake text')
word=df[df['label']==0]['text'].str.split().apply(lambda x : [len(i) for i in x])
sns.distplot(word.map(lambda x: np.mean(x)),ax=ax2,color='green')
ax2.set_title('Not Fake text')
fig.suptitle('Average word length in each text')

X = df['text'].values
y = df['label'].values

print(X)

## TFidf Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer(max_features=5000,ngram_range=(1,3))
vec.fit(X)
X_vec = vec.transform(X)
X_vec

X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.33, random_state=0,stratify=y)

print("X train Shape :", X_train.shape)
print("X test Shape :", X_test.shape)
print("y train Shape :", y_train.shape)
print("y test Shape :", y_test.shape)

# tfidf features

vec.get_feature_names()[:50]

from sklearn.naive_bayes import MultinomialNB

mnb =MultinomialNB()
model = mnb.fit(X_train,y_train)


pickle.dump(best_model, open('best_model.pkl', 'wb'))

from sklearn import metrics
pred =model.predict(X_test)
score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)
cm = metrics.confusion_matrix(y_test, pred)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    See full source and example: 
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
plot_confusion_matrix(cm,classes=['Fake', 'Not Fake'])

# Test on test data


def predict_fakenews(news):
    news= [news]
    news_vec = vec.transform(news)
    pred = model.predict(news_vec)
    return pred

import random
sample_text = random.choice(df[["text",'label']].values.tolist())
prediction = predict_fakenews(sample_text[0])
prediction  = int(prediction)
if prediction == 0 :
    prediction = 'not fake'
else:
    prediction = 'fake'
print('The predicted label of the news is {}: '.format(prediction))
print('The actual label of the news is is {}: '.format(sample_text[1]))


import pickle

pickle.dump(model, open('/Users/kush/Downloads/Fake News Classifier/mnb_model.pkl', 'wb'))

pickle.dump(vec, open('/Users/kush/Downloads/Fake News Classifier/tfidf_vec.pkl', 'wb'))
