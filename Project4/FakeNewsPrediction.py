import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords   #nltk---> natural language tool kit
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
import nltk
nltk.download('stopwords')

# print(stopwords.words('english'))     #printing the stopwords

#data preprocessing
news_dataset = pd.read_csv('/Users/sthaarekh/Documents/ /                     /Python/Project4/train.csv')
news_dataset.shape
news_dataset.head()
news_dataset.isnull().sum()
news_dataset = news_dataset.fillna('')      #replacing the null values with null string



