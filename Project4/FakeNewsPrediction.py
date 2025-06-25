import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords   #nltk---> natural language tool kit
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk
nltk.download('stopwords')

# print(stopwords.words('english'))     #printing the stopwords

#data preprocessing
news_dataset = pd.read_csv('/Users/sthaarekh/Documents/ /                     /Python/Project4/train.csv')
news_dataset.shape
news_dataset.head()
news_dataset.isnull().sum()
news_dataset = news_dataset.fillna('')      #replacing the null values with null string

#merging the author and title
news_dataset['content'] = news_dataset['author']+' '+ news_dataset['title']
print(news_dataset['content'])

#separating the data and label
x = news_dataset.drop('label', axis=1)
y = news_dataset['label']
print(x,y)

#Stemming: The process of reducing a word to its root word --> example: actor, acting, actress -> act
port_stem = PorterStemmer()
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

news_dataset['content'] = news_dataset['content'].apply(stemming)
print(news_dataset['content'])

#defining new data and label
x = news_dataset['content'].values
y = news_dataset['label'].values

#converting the textual data to numerical data
vectorizer = TfidfVectorizer()
vectorizer.fit(x)
x = vectorizer.transform(x)
print(x)

#splitting into train and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)

#training the model
model = LogisticRegression()
model.fit(x_train, y_train)

#prediction on train data
x_train_prediction = model.predict(x_train)
accuracy_train = accuracy_score(x_train_prediction, y_train)
print(accuracy_train)

#prediction on test data
x_test_prediction = model.predict(x_test)
accuracy_test = accuracy_score(x_test_prediction, y_test)
print(accuracy_test)

#Making a predictive system
x_new = x_test[0]
predicted = model.predict(x_new)
if(predicted==0):
    print("This news is real")
else:
    print("This news is fake")
    