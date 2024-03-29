import tensorflow as tf 
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

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

# Remove rows where either 'Comment' or 'sentiment' column contains null values
df.dropna(subset=['Comment', 'sentiment'], inplace=True)

# Convert the sentiment column to numeric values
sentiment_mapping = {'positive': 1, 'negative': -1, 'neutral': 0}
df['sentiment'] = df['sentiment'].map(sentiment_mapping)

# Removing duplicates
df.drop_duplicates(inplace=True)

# Stemming the data
port_stem = PorterStemmer()

def stemming(content):
  stemmed_content = re.sub('[^a-zA-Z]',' ',content)
  stemmed_content = stemmed_content.lower()
  stemmed_content = stemmed_content.split()
  stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
  stemmed_content = ' '.join(stemmed_content)
  return stemmed_content

#stemming contents of the comment section
df['stemmed_content'] = df['Comment'].apply(stemming)
print(df['stemmed_content'])
