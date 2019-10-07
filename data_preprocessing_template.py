import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv("Data.csv")
X=dataset.iloc[:, :-1].values
Y=dataset.iloc[:, 3].values

#taking cre of missig values


from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
imputer=imputer.fit(X[:, 1:3])
X[:, 1:3]=imputer.transform(X[:, 1:3])

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
lableencoder_X=LabelEncoder()
X[:, 0]=lableencoder_X.fit_transform(X[:, 0])

onehotencoder=OneHotEncoder(handle_unknown='ignore',sparse=False,categorical_features=[0])
X=onehotencoder.fit_transform(X)

lableencoder_Y=LabelEncoder()
Y=lableencoder_Y.fit_transform(Y)