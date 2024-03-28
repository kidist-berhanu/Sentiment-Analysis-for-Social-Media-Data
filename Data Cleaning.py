import tensorflow as tf
import numpy as np
import pandas as pd

file_path = r'C:\Users\Yabsra\Desktop\Projects\Selected topics\Sentiment-Analysis-for-Social-Media-Data\Gaming_comments_sentiments_from_Reddit(Dataset).csv'

df = pd.read_csv(file_path)

# Data Cleaning

# Remove any rows with missing values
df.dropna(inplace=True)

