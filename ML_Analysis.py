## Import necessary packages
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import string
import re
from nltk.corpus import stopwords

import numpy as np
from math import exp, log, sqrt
import datetime
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.express as px

## Check Data style, 1 column Articles titles 1 column sentiment view Positive, Negative, Neutral
data = pd.read_csv("all-data.csv")

## First Process Tokenization of Articles:

## package library download first code running
##nltk.download('punkt') ## Download To do during first running code
##nltk.download('stopwords')

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

print(cleaned_title(text))