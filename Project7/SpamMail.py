import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#data collection and preprocessing
mail_data = pd.read_csv('/Users/sthaarekh/Documents/ /                     /Python/Project7/mail_data.csv')
print(mail_data.tail())

# label spam as 0 and ham as 1
mail_data.loc[mail_data['Category']== 'spam', 'Category',] = 0
mail_data.loc[mail_data['Category']== 'ham', 'Category',] = 1

#splitting the category and messages
x = mail_data['Message']
y = mail_data['Category']

y = y.astype('int')

#training and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)

#tranforming the data into feature vectors
feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english', lowercase = True)

x_train_features = feature_extraction.fit_transform(x_train)
x_test_features = feature_extraction.transform(x_test)
print(x_train_features)