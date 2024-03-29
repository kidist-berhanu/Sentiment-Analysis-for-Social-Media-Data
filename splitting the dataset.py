import keras
import nltk
import pandas as pd
import numpy as np
import re
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
from numpy import array
from tensorflow.keras.preprocessing.text import one_hot, Tokenizer
from keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Dense
from keras.layers import Flatten, GlobalMaxPooling1D, Embedding, Conv1D, LSTM
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
#read dataser
df = pd.read_csv("C:/Users/HP/Documents/Gaming_comments_sentiments on reddit.csv")
#show the first 5 datas in the dataset
df.head()
#shows row and columns
df.shape
# draws graph
import seaborn as sns
sns.countplot(x='sentiment', data=df)
#
df["Comment"][4]


