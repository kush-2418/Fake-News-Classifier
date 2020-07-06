#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 14:43:02 2020

@author: kush
"""


import streamlit as st
import pickle
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
import re,string
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
punctuation = list(string.punctuation)
stop_words.update(punctuation)# Data Cleaning
ps = PorterStemmer()
lemma = WordNetLemmatizer()

def main():
    def load_models():
        model = pickle.load(open('/Users/kush/Downloads/Fake News Classifier/mnb_model.pkl', 'rb'))
        vec = pickle.load(open('/Users/kush/Downloads/Fake News Classifier/tfidf_vec.pkl', 'rb'))

        return model,vec
    
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
   
    model,vec = load_models()

    st.markdown("<body style='background-color:orange;'><h1 style='text-align: center; color: white;'>Fake News Classifier</h1></body>", unsafe_allow_html=True)
    st.markdown("<body style='background-color:white;'><h2 style='text-align: center; color: blue;'> Predict on News</h2></body>", unsafe_allow_html=True)
    st.markdown("<body style='background-color:white;'><h3 style='text-align: center; color: red;'> Enter the text to know whether it's Fake or Real 👇</h3></body>", unsafe_allow_html=True)
    news = st.text_input("")

    if st.button('Predict'):
        news = clean_text(news)
        news = [news]
        news_vec = vec.transform(news).toarray()
        pred = model.predict(news_vec)

        if pred == 0:
            st.error("Ohh!!! It is a Fake News 😟")
        else:
            st.info("Don't Worry this News is Real 🙂")
            st.balloons()
    
    
    


if __name__ == '__main__':
    main()