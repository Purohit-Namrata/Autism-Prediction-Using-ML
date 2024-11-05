import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from sklearn.feature_extraction.text import TfidfVectorizer

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('C:/Users/BLAUPLUG/Documents/Python_programs/Autism Prediction using ML/data_csv.csv')
print(df.head())
print(df.shape)

#print(df.info())

print(df.describe())

print(df['ASD_traits'].value_counts())

df.loc[df['ASD_traits']=='No','ASD_traits',]=0
df.loc[df['ASD_traits']=='Yes','ASD_traits',]=1

print(df['ASD_traits'].value_counts())

X=df.drop(columns="ASD_traits",axis=1)
y=df["ASD_traits"]

print(X)
print(y)

scaler=StandardScaler()
scaler.fit(X)

standaridized_data=scaler.transform(X)
print(standaridized_data)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2)
print(X.shape, X_train.shape, X_test.shape)

model=linear_model.LogisticRegression()
model.fit(X_train,y_train)

X_train_prediction=model.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,y_train)
print("Accuracy score of training data ",training_data_accuracy)

X_test_prediction=model.predict(X_test)
test_data_accuracy=accuracy_score(X_test_prediction,y_test)
print("Accuracy score of test data ",test_data_accuracy)


