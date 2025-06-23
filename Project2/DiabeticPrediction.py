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