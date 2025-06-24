import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics


#importing the dataset of california
house_dataset = fetch_california_housing()
print(house_dataset)

#loading the dataset into pandas
house_dataframe = pd.DataFrame(house_dataset.data, columns = house_dataset.feature_names)
house_dataframe['price'] = house_dataset.target
print(house_dataframe)

#understanding the correlation between features
correlation = house_dataframe.corr()

#constructing a heatmap
plt.figure(figsize = (10,10))
sns.heatmap(correlation, cbar=True, square=True, fmt='.2f', annot=True, annot_kws={'size':8}, cmap='Blues')
plt.show()

#splitting the data and target
x = house_dataframe.drop('price', axis = 1)
y = house_dataframe['price']

#splitting the training and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
print(x_train.shape, x_test.shape)

#model training
model = XGBRegressor()

model.fit(x_train, y_train)

#prediction on training and finding error
x_train_prediction = model.predict(x_train)

#r2 error
r2_train = metrics.r2_score(y_train, x_train_prediction)
print(r2_train)

#absolute mean error
abs_train = metrics.mean_absolute_error(y_train, x_train_prediction)
print(abs_train)


#prediction on testing and finding error
x_test_prediction = model.predict(x_test)

#r2 error
r2_test = metrics.r2_score(y_test, x_test_prediction)
print(r2_test)

#absolute mean error
abs_test = metrics.mean_absolute_error(y_test, x_test_prediction)
print(abs_test)