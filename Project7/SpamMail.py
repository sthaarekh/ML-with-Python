import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import TfidVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#data collection and preprocessing
mail_data = pd.read_csv('/Users/sthaarekh/Documents/ /                     /Python/Project7/mail_data.csv')
print(mail_data.tail())

# label spam as 0 and ham as 1
mail_data.loc[mail_data['Category']== 'spam', 'Category',] = 0
mail_data.loc[mail_data['Category']== 'ham', 'Category',] = 1
