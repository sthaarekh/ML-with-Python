import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

#importing dataset
diabetes_data = pd.read_csv('/Users/sthaarekh/Documents/ /                     /Python/Project2/diabetes.csv')
print(diabetes_data.shape)
print(diabetes_data.describe())
print(diabetes_data['Outcome'].value_counts())

#splitting the data
x = diabetes_data.drop('Outcome', axis=1)
y = diabetes_data['Outcome']

#standarizing the data
scaler = StandardScaler()
scaler.fit(x)
standerized_data = scaler.transform(x)
print(standerized_data)
x = standerized_data

#traning and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)

#training the model
classifier = svm.SVC(kernel = 'linear')
classifier.fit(x_train, y_train)

#training data accuracy
x_train_prediction = classifier.predict(x_train)
accuracy_train = accuracy_score(x_train_prediction, y_train)
print("Training accuracy score is", accuracy_train)

#testing data accuracy
x_test_prediction = classifier.predict(x_test)
accuracy_test = accuracy_score(x_test_prediction, y_test)
print("Testing accuracy score is", accuracy_test)