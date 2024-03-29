import tensorflow as tf 
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords

file_path = r'Gaming_comments_sentiments_from_Reddit(Dataset).csv'

df = pd.read_csv(file_path)

# Data Cleaning

# Remove any rows with missing values
df.dropna(inplace=True)

# Convert the sentiment column to numeric values
sentiment_mapping = {'positive': 1, 'negative': -1, 'neutral': 0}
df['sentiment'] = df['sentiment'].map(sentiment_mapping)

# Removing duplicates
df.drop_duplicates(inplace=True)
#to check the structure of the DataFrame
print(df.info())
