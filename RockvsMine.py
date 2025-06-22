import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

sonar_data = pd.read_csv('/Users/sthaarekh/Documents/ /                     /Python/sonar.csv', header=None)
print(sonar_data.describe())