#import libraries
import pandas as pd
import numpy as np
import pickle 
#import matplotlib.pyplot as plt
#from sklearn import *
#import data
df = pd.read_csv('cleveland.csv')
#te = pd.read_csv('input.csv')
#Adding columns
df.columns = ["age", "sex", "cp", "restbp", "chol", "fbs", "restecg", 
           "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
#te.columns = ["age", "sex", "cp", "restbp", "chol", "fbs", "restecg", 
#           "thalach", "exang", "oldpeak", "slope", "ca", "thal"]                                             

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
print(X_train)
print(" -----------------------------------------")
#Standard scaling data
from sklearn.preprocessing import StandardScaler 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
pickle.dump(scaler, open('scaler.pkl','wb'))

#Logistic Regression Algorithm
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()

log_reg.fit(X_train, y_train)
print(y_train)
print("bvbfvbvdwibfpiwbvenvbfioehfbgirejfnbvifjvbfdfofnbvgjfbfnergbferjgbegfbg")
print(y_test)
print(log_reg.score(X_train,y_train))
print(log_reg.score(X_test, y_test))

y_pred_log_reg = log_reg.predict(X_test)
print(y_pred_log_reg)
#print(te)
check = np.array([[49,1,2,130,266,0,0,171,0,0.6,1,0,3]])
#sc = pickle.load(open('scaler.pkl','rb'))
print(scaler.mean_)
for i in range(12):
    print(check[0][i])
    check[0][i] = check[0][i] - scaler.mean_[i]
print("---------------------------------------------")
print(check)    
#s = scaler.fit_transform(check)
#print(s)


# = sc.fit_transform(te)
#print(te)
print(log_reg.predict(check))
#from sklearn.neighbors import KNeighborsClassifier

#knn=KNeighborsClassifier(n_neighbors=29)
#knn.fit(X_train , y_train)

#print(knn.score(X_train,y_train))
#print(knn.score(X_test, y_test))

#knnn = knn.predict(X_test)


#print(knn.predict([[58,0,4,130,197,0,0,131,0,0.6,2,0,3]]))

pickle.dump(log_reg,open('log_model.pkl','wb'))
