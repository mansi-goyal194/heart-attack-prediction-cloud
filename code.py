#import libraries
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#from sklearn import *
#import data
df = pd.read_csv('cleveland.csv')

#Adding columns
df.columns = ["age", "sex", "cp", "restbp", "chol", "fbs", "restecg", 
           "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
                                               

#print(df.to_string())
df['target'] = df.target.map({0: 0, 1: 1, 2: 1, 3: 1, 4: 1})
#print(df.to_string())
#handling null value
df['thal'] =df.thal.fillna(df.thal.mean())
df['ca'] = df.ca.fillna(df.ca.mean())

#seprating feature matrix and feature vector
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values


#Seprating train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Standard scaling data
from sklearn.preprocessing import StandardScaler as ss
sc = ss()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#Logistic Regression Algorithm
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()

log_reg.fit(X_train, y_train)
print(y_train)
print("bvbfvbvdwibfpiwbvenvbfioehfbgirejfnbvifjvbfdfofnbvgjfbfnergbferjgbegfbg")
print(y_test)
print(log_reg.score(X_train,y_train))
print(log_reg.score(X_test, y_test))

#y_pred_log_reg = log_reg.predict(X_test)

print(log_reg.predict([[56,1,2,130,221,0,2,163,0,0,1,0,7]]))
#from sklearn.neighbors import KNeighborsClassifier

#knn=KNeighborsClassifier(n_neighbors=29)
#knn.fit(X_train , y_train)

#print(knn.score(X_train,y_train))
#print(knn.score(X_test, y_test))

#knnn = knn.predict(X_test)


#print(knn.predict([[58,0,4,130,197,0,0,131,0,0.6,2,0,3]]))
import pickle 
pickle.dump(log_reg,open('log_model.pkl','wb'))