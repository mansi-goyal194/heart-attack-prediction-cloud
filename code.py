#import libraries
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from sklearn import *
#import data
df = pd.read_csv('cleveland.csv')

#Adding columns
df.columns = ["age", "sex", "cp", "restbp", "chol", "fbs", "restecg", 
           "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]



df['target'] = df.target.map({0: 0, 1: 1, 2: 1, 3: 1, 4: 1})

#handling null value
df['thl'] =df.thal.fillna(df.thal.mean())
df['ca'] = df.ca.fillna(df.ca.mean())

#seprating feature matrix and feature vector
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values


#Seprating train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Standard scaling data
#from sklearn.preprocessing import StandardScaler as ss
#sc = ss()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)


#Logistic Regression Algorithm
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()

log_reg.fit(X_train, y_train)

import pickle 
pickle.dump(log_reg,open('log_model.pkl','wb'))