## Import necessary packages
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import string
import re
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import numpy as np
from math import exp, log, sqrt
import datetime
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.express as px

## Sklearn packages extractions
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


## Check Data style, 1 column Articles titles 1 column sentiment view Positive, Negative, Neutral
data = pd.read_csv("all-data.csv")



# Sample text to tokenize
text = data.iloc[1,1]


def cleaned_title(article):
    # Tokenize the text
    tokens = word_tokenize(text)

    # Convert tokens to lowercase
    tokens_lower = [token.lower() for token in tokens]

    # Remove punctuation
    table = str.maketrans('', '', string.punctuation)
    tokens_no_punct = [token.translate(table) for token in tokens_lower]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens_no_stop = [token for token in tokens_no_punct if token not in stop_words]

    # Remove special characters
    tokens_cleaned = [re.sub(r'[^a-zA-Z0-9]', '', token) for token in tokens_no_stop if token.strip()]
    return tokens_cleaned

token = cleaned_title(text)
print(token)


## Split artciles as texts and sentiment as labels
texts = data.iloc[1:len(data),1]
labels = data.iloc[1:len(data),0]

## Initialize TF-IDF vectorizer
vectorizer = TfidfVectorizer()
## Fit the vectorizer to the input example and transform the data into TF-IDF vectors
X = vectorizer.fit_transform(texts)
## Get the feature names (terms in the vocabulary)
feature_names = vectorizer.get_feature_names()

## Initialize label encoder
label_encoder = LabelEncoder()
## Encode the labels
encoded_labels = label_encoder.fit_transform(labels)



# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, encoded_labels, test_size=0.2, random_state=42)

# Train a classification model (e.g., Logistic Regression)
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Example of predicting sentiment on new data
new_text = ["Hasbro laid off people following major losses on the markets"]
new_text_vectorized = vectorizer.transform(new_text)
predicted_sentiment = model.predict(new_text_vectorized)
predicted_sentiment_str = label_encoder.inverse_transform(predicted_sentiment)
print("Predicted sentiment:", predicted_sentiment_str)