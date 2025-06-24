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