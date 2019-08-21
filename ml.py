# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 16:47:01 2019

@author: Rohan
"""

import pandas as pd

#Dividing Dataset
dataset=pd.read_csv('Social_Network_Ads.csv')
X=dataset.iloc[:,2:-1].values
Y=dataset.iloc[:,-1].values



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

