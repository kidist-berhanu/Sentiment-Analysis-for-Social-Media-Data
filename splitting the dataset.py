import tensorflow as tf
import numpy as np
import pandas as pd

file_path = r'C:\Users\Yabsra\Desktop\Projects\Selected topics\Sentiment-Analysis-for-Social-Media-Data\Gaming_comments_sentiments_from_Reddit(Dataset).csv'

df = pd.read_csv(file_path)

# Data Cleaning

# Remove any rows with missing values
df.dropna(inplace=True)

# Convert the sentiment column to numeric values
sentiment_mapping = {'positive': 1, 'negative': -1, 'neutral': 0}
df['sentiment'] = df['sentiment'].map(sentiment_mapping)

#splitting data in to training and testing
from sklearn.model_selection import train_test_split

features = df['comment'] 
labels = df['sentiment']

# (80% for training, 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
train_data=train_data.apply(lambda features:features.split())
test_data=test_data.apply(lambda labels:comment.split())
