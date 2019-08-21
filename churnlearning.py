# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 12:41:04 2019

@author: User
"""

import pandas as pd

dataset=pd.read_csv('Churn_Modelling.csv')

#Dividing Dataset
X=dataset.iloc[:,3:-1].values
Y=dataset.iloc[:,-1].values

#encoding categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
lb_X=LabelEncoder()
X[:,1]=lb_X.fit_transform(X[:,1])
X[:,2]=lb_X.fit_transform(X[:,2])

oneh=OneHotEncoder(categorical_features=[1])
X=oneh.fit_transform(X).toarray()

lb_Y=LabelEncoder()
Y=lb_Y.fit_transform(Y)

#train test split
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)

#scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.fit_transform(X_test)

#Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(X_train,Y_train)

#predict
Y_pred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,Y_pred)



