import tensorflow as tf 
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


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

# Visualize Data
import matplotlib.pyplot as plt
# Pie chart
# Counting occurrences of each sentiment
sentiment_counts = df['sentiment'].value_counts()
sentiment_counts = sentiment_counts.rename({1: 'Positive', -1: 'Negative', 0: 'Neutral'})

# Plotting a pie chart
plt.figure(figsize=(6, 6))
plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
plt.title('Sentiment Distribution')
plt.show()

# splitting the data to training data and test data
X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size = 0.2, stratify = Y, random_state = 2)
print(X.shape, X_train.shape, X_test.shape)

# converting the textual data to numerical data
vectorizer = TfidfVectorizer()
X_train =  vectorizer.fit_transform(X_train)
X_test =  vectorizer.transform(X_test)

# Training the model using logistic regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train,Y_train)

# Model Evaluation 
# Accuracy score on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print(training_data_accuracy)

# Accuracy score on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print(test_data_accuracy)
