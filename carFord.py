# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 21:45:49 2021

@author: user
"""


import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# Data with salary and ID
file = pd.read_csv("F:/ML Datasets/Ford.csv")

#a  = file.isnull().sum()

x1 = file.iloc[: ,0:2]
x2 = file.iloc[: , 3:]
x =pd.concat([x1 ,x2] , axis=1)

y = file.iloc[:,2].values.reshape(-1,1)
'''
label_encode = []
onehot_encode = []

for i in len(range(x.columns)):
    if(len(x.iloc[:,i].valuecounts()) == 2):
        label_encode.append(i)
        
    
    elif(len(x.iloc[:,i].valuecounts()) > 2):
        onehot_encode.append(i)
'''

num_indices = []
cate_indices = []


for i in range(len(x.columns)):
    if(x.iloc[:,i].dtype == 'O'):
        cate_indices.append(i)
    elif(x.iloc[:,i].dtype == "int64" or x.iloc[:,i].dtype == "float64"):
        num_indices.append(i)
    
cate_label_indices = []
cate_onehot_indices = []

for i in cate_indices:
    if(len(x.iloc[:,i].value_counts())==2):
        cate_label_indices.append(i)
        
    elif(len(x.iloc[:,i].value_counts()) > 2):
        cate_onehot_indices.append(i)


ct  = ColumnTransformer(transformers=[('encoder' , OneHotEncoder() , cate_onehot_indices)],remainder='passthrough')
x = ct.fit_transform(x)
x = x.toarray()

from sklearn.model_selection import train_test_split
x_train , x_test , y_train ,y_test = train_test_split(x , y ,test_size=0.2 , random_state=0)

from sklearn.svm  import SVR
regressor = SVR(kernel='rbf')
regressor.fit(x_train , y_train)
y_pred = regressor.predict(x_test)


y_pred = y_pred.reshape(len(y_pred),1)
y_test = y_test.reshape(len(y_test),1)

concat = np.concatenate((y_pred,y_test),axis=1)
