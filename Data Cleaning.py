import tensorflow as tf 
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords

nltk.download('stopwords')

file_path = r'Gaming_comments_sentiments_from_Reddit(Dataset).csv'
df = pd.read_csv(file_path)

# Data Cleaning

# Remove stop words
stop_words = set(stopwords.words('english'))
df['Comment'] = df['Comment'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stop_words]))
df['sentiment'] = df['sentiment'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stop_words]))

# Convert the sentiment column to numeric values
sentiment_mapping = {'positive': 1, 'negative': -1, 'neutral': 0}
df['sentiment'] = df['sentiment'].map(sentiment_mapping)

# Removing duplicates
df.drop_duplicates(inplace=True)
#to check the structure of the DataFrame
print(df.info())
