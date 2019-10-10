# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 10:23:11 2019

@author: neeratig
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#importing dataset
dataset=pd.read_csv('50_Startups.csv')
#spilting independent and dependent variable
X=dataset.iloc[:, :-1].values
y=dataset.iloc[:, 4].values
#encoding categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
lableencoder=LabelEncoder()
X[:, 3]=lableencoder.fit_transform(X[:, 3])
onehotencoder=OneHotEncoder(categorical_features=[3])
X=onehotencoder.fit_transform(X).toarray()
#avoiding dummy variable trap
#X=X[:, 1:]
#spiting data into training and testing set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
#fitting multiple linear regression to training set
from sklearn.linear_model import LinearRegression
regress=LinearRegression()
regress.fit(X_train,y_train)
#predicting test set result
y_predict=regress.predict(X_test)



