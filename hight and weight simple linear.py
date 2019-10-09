# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 16:17:01 2019

@author: neeratig
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('data.csv')
X=dataset.iloc[:, :-1]
y=dataset.iloc[:, 1]

#spilting traing set and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)
#fitting linear regression to training set
from sklearn.linear_model import LinearRegression
regress=LinearRegression()
regress.fit(X_train,y_train)

#visualize training set
viz_test=plt
viz_test.scatter(X_train,y_train,c='red')
viz_test.plot(X_train,regress.predict(X_train),color='blue')
viz_test.title('height vs weight(training set)')
viz_test.xlabel('height')
viz_test.ylabel('weight')
viz_test.show()
#visualize test set
viz_test = plt
viz_test.scatter(X_test, y_test, color='red')
viz_test.plot(X_train,regress.predict(X_train),color='blue')
viz_test.title('height VS wieght (Test set)')
viz_test.xlabel('hight')
viz_test.ylabel('wieght')
viz_test.show()

#prdicting weight
y_predict=regress.predict(X_test)

#predicting single value
weight_pre=np.array(1.5).reshape(-1,1)
regress.predict(weight_pre)




