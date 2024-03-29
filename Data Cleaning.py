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

# Remove special characters
df['Comment'] = df['Comment'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x))
df['sentiment'] = df['sentiment'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x))
