import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#data collection and preprocessing
movies_data = pd.read_csv('/Users/sthaarekh/Documents/ /                     /Python/Project8/movies.csv')
# print(movies_data.head())
# print(movies_data.isnull().sum())

#selecting the relevant features
selected_features = ['genres', 'keywords', 'overview', 'tagline', 'cast', 'director']

#replacing the missing values
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

combined_features = movies_data['genres'] + ' ' + movies_data['keywords'] +  movies_data['overview'] +' ' + movies_data['tagline']+ ' ' + movies_data['cast']+ ' ' + movies_data['director']

#converting the text data to feature vectors
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)

#finding the cosine similarity
similarity = cosine_similarity(feature_vectors)
print(similarity.shape)

#taking movie name from user
movie_name = input('Enter your favourite movie name: ')

#creating a  list of all movie name
list_of_movies = movies_data['title'].tolist()

#finding the closest match to the input
find_close_match = difflib.get_close_matches(movie_name, list_of_movies)

closest_match = find_close_match[0]

#finding the index of the movie
index = movies_data[movies_data.title == closest_match]['index'].values[0]
print(index)

#getting a list of similar movies
similarity_score = list(enumerate(similarity[index]))
print(similarity_score)

#soritng the movies from highest similarity score
sorted_list = sorted(similarity_score, key= lambda x:x[1], reverse=True)
print(sorted_list)

#printing the similar movies
i = 1
for movies in sorted_list:
    index = movies[0]
    title_of_movie = movies_data[movies_data.index == index]['title'].values[0]
    if(i<=20):
        print(i,". ",title_of_movie)
        i+=1
