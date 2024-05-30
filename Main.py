# Importing necessary libraries
import tensorflow as tf 
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download the stopwords from nltk
nltk.download('stopwords')

# Define the file path for the dataset
file_path = r'/Gaming_comments_sentiments_from_Reddit(Dataset).csv'

# Load the dataset into a pandas DataFrame
df = pd.read_csv(file_path)

# Data Cleaning

# Remove stop words from 'Comment' and 'sentiment' columns
stop_words = set(stopwords.words('english'))
df['Comment'] = df['Comment'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stop_words]))
df['sentiment'] = df['sentiment'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stop_words]))

# Remove special characters from 'Comment' and 'sentiment' columns
df['Comment'] = df['Comment'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x))
df['sentiment'] = df['sentiment'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x))

# Remove rows where either 'Comment' or 'sentiment' column contains null values
df.dropna(subset=['Comment', 'sentiment'], inplace=True)

# Convert the sentiment column to numeric values using a mapping dictionary
sentiment_mapping = {'positive': 1, 'negative': -1, 'neutral': 0}
df['sentiment'] = df['sentiment'].map(sentiment_mapping)

# Remove duplicate rows from the DataFrame
df.drop_duplicates(inplace=True)

# Initialize the PorterStemmer for stemming the data
port_stem = PorterStemmer()

# Define a function for stemming content
def stemming(content):
    # Remove non-alphabetic characters, convert to lowercase, split into words
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    # Stem each word and join them back into a single string
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

# Apply stemming to the 'Comment' column and create a new column 'stemmed_content'
df['stemmed_content'] = df['Comment'].apply(stemming)

# Print the stemmed content to verify the result
print(df['stemmed_content'])

# Define the file path for the cleaned data
cleaned_file_path = r'/cleaned_data.csv'

# Save the cleaned DataFrame to a new CSV file
df.to_csv(cleaned_file_path, index=False)
