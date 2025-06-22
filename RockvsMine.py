import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

sonar_data = pd.read_csv('/Users/sthaarekh/Documents/ /                     /Python/sonar.csv', header=None)

print(sonar_data.describe())  # gives the overview of the data

#separating data and labels
x = sonar_data.drop(columns=60, axis=1)
y = sonar_data[60]  #--> stores the data of rock and mine
print(x,y)

#splitting training and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, stratify=y, random_state=1)
print(x.shape, x_train.shape, x_test.shape)