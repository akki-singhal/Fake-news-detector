#Library Required

import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

#Dataset Import
news_df = pd.read_csv('Dataset check.csv')

#Dataset Edit
news_df.isna().sum()
news_df = news_df.fillna(' ')

#Stemming
ps = PorterStemmer()
def stemming(title):
    stemmed_title = re.sub('[^a-zA-Z]',' ',title)
    stemmed_title = stemmed_title.lower()
    stemmed_title = stemmed_title.split()
    stemmed_title = [ps.stem(word) for word in stemmed_title if not word in stopwords.words('english')]
    stemmed_title = " ".join(stemmed_title)
    return stemmed_title

news_df['title'] = news_df['title'].apply(stemming)

#Vectorize data
X = news_df['title'].values
Y = news_df['label'].values
vector = TfidfVectorizer()
vector.fit(X)
X = vector.transform(X)

#data into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,stratify=Y, random_state=1)

#Model Implementation
model = LogisticRegression()
model.fit(X_train,Y_train)

#Website
st.title("Fake News Detector")
input_text=st.text_input("Enter News Article")

def prediction(input_text):
    input_data=vector.transform([input_text])
    prediction = model.predict(input_data)
    return prediction[0]

if input_text:
    pred = prediction(input_text)
    if pred == 1:
        st.write("News is Fake")
    else:
        st.write("News is Real")