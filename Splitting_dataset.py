import tensorflow as tf 
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

nltk.download('stopwords')

file_path = r'/Gaming_comments_sentiments_from_Reddit(Dataset).csv'
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

# Define the file path for the cleaned data
cleaned_file_path = r'/cleaned_data.csv'

# Save the cleaned DataFrame to a new CSV file
df.to_csv(cleaned_file_path, index=False)

# separating the comment and sentiment
X = df['stemmed_content'].values
Y = df['sentiment'].values

# splitting the data to training data and test data
X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size = 0.2, stratify = Y, random_state = 2)
print(X.shape, X_train.shape, X_test.shape)

# converting the textual data to numerical data
vectorizer = TfidfVectorizer()
X_train =  vectorizer.fit_transform(X_train)
X_test =  vectorizer.transform(X_test)
