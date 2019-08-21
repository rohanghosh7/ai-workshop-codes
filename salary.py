# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 11:21:45 2019

@author: User
"""

import pandas as pd

dataset=pd.read_csv('salary.csv')

#Dividing Dataset
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values

#train test split
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)

#linear Regression
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)

Y_pred=regressor.predict(X_test)

import matplotlib.pyplot as plt

plt.scatter(X_train,Y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs Years of Exp')
plt.xlabel('Yrs of Exp')
plt.ylabel('Salary')
plt.show()

plt.scatter(X_test,Y_test,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs Years of Exp')
plt.xlabel('Yrs of Exp')
plt.ylabel('Salary')
plt.show()


































































































































































































































































































































































































































































































































