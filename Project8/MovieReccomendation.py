import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#data collection and preprocessing
movies_data = pd.read_csv('/Users/sthaarekh/Documents/ /                     /Python/Project8/movies.csv')
print(movies_data.head())
print(movies_data.isnull().sum())

#selecting the relevant features
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']

#replacing the missing values
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

combined_features = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + movies_data['tagline']+ ' ' + movies_data['cast']+ ' ' + movies_data['director']

#converting the text data to feature vectors
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)

#finding the cosine similarity
similarity = cosine_similarity(feature_vectors)
print(similarity.shape)