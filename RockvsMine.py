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

#Model training
model = LogisticRegression()  #---> using linear regression model
model.fit(x_train, y_train)

#Model accuracy
x_train_prediction = model.predict(x_train)
accuracy_train = accuracy_score(x_train_prediction, y_train)
print(accuracy_train)

x_test_prediction = model.predict(x_test)
accuracy_test = accuracy_score(x_test_prediction, y_test)
print(accuracy_test)

#now testing for the individual data and getting the prediction
input_data = (0.0968,0.0821,0.0629,0.0608,0.0617,0.1207,0.0944,0.4223,0.5744,0.5025,0.3488,0.1700,0.2076,0.3087,0.4224,0.5312,0.2436,0.1884,0.1908,0.8321,1.0000,0.4076,0.0960,0.1928,0.2419,0.3790,0.2893,0.3451,0.3777,0.5213,0.2316,0.3335,0.4781,0.6116,0.6705,0.7375,0.7356,0.7792,0.6788,0.5259,0.2762,0.1545,0.2019,0.2231,0.4221,0.3067,0.1329,0.1349,0.1057,0.0499,0.0206,0.0073,0.0081,0.0303,0.0190,0.0212,0.0126,0.0201,0.0210,0.0041)

#changing the input data to numpy array
input_data_as_array = np.asarray(input_data)

#reshape the numpy array 
input_data_reshaped = input_data_as_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)
