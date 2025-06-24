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