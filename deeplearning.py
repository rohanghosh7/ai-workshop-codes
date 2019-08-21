# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 14:36:56 2019

@author: rohanghosh
"""


import pandas as pd

dataset=pd.read_csv('Churn_Modelling.csv')
X=dataset.iloc[:,3:-1].values
Y=dataset.iloc[:,-1].values

#encoding categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
lb_X_1=LabelEncoder()
X[:,1]=lb_X_1.fit_transform(X[:,1])
lb_X_2=LabelEncoder()
X[:,2]=lb_X_2.fit_transform(X[:,2])

oneh=OneHotEncoder(categorical_features=[1])

X=oneh.fit_transform(X).toarray()



#train test split
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)

#scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.fit_transform(X_test)

#ANN

import keras
from keras.models import Sequential
from keras.layers import Dense

classifier=Sequential()

#Add input and hidden layer
classifier.add(Dense(6,kernel_initializer='uniform',activation='relu',input_dim=12))

#Add second hidden layer
classifier.add(Dense(6,kernel_initializer='uniform',activation='relu'))

#Add output layer
classifier.add(Dense(1,kernel_initializer='uniform',activation='sigmoid'))

#Compile
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#Training
classifier.fit(X_train,Y_train,batch_size=10,epochs=100)

Y_pred=classifier.predict(X_test)
Y_pred=(Y_pred>0.5)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,Y_pred)

