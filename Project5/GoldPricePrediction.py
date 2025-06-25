import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

#loading the gold data
gold_data = pd.read_csv('/Users/sthaarekh/Documents/ /                     /Python/Project5/gold_model_dataset_2015_2025.csv')
gold_data = gold_data.drop(index=gold_data.index[0])
print(gold_data)
#data preprocessing
gold_data.info()
gold_data.isnull().sum()
gold_data.describe()

#finding correlation among the data
correlation = gold_data.select_dtypes(include='number').corr()
#constructing a heatmap
plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar=True, square=True, fmt='.2f', annot=True, annot_kws={'size':8}, cmap='Blues')
plt.show()

#range of GLD value
sns.displot(gold_data['GLD'], color='Green')
plt.show()

#splitting the data values
x = gold_data.drop(['Date', 'GLD'], axis = 1)
y = gold_data['GLD']

#training and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
print(x_train.shape, x_test.shape)

#model training
model = RandomForestRegressor()

model.fit(x_train, y_train)

#model testing 
x_test_prediction = model.predict(x_test)
print(x_test_prediction)

#model error score
#r2 error
r2 = metrics.r2_score(y_test, x_test_prediction)
print(r2)

#comparing the actual and predicted value
y_test = list(y_test)
plt.plot(y_test, color='blue', label = 'Actual Value')
plt.plot(x_test_prediction, color='green', label='Predicted Value')
plt.title('Actual Price vs Predicted Price')
plt.xlabel('Number of values')
plt.ylabel('GLD Price')
plt.legend()
plt.show()

#predicting the gold price
input_data = (6092.18, 73.00, 32.61, 1.1618)
input_data_asarray = np.asarray(input_data)
input_data_reshaped = input_data_asarray.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print("The predicted gold price is",prediction[0])