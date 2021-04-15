# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 20:31:31 2021

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 13:31:03 2021

@author: user
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


data = pd.read_csv('bmw.csv')

null = data.isnull().sum()

# Select all columns except price
x = data[data.columns.difference(['price'])]
#Dependent variable price
y = data['price'].values.reshape(-1,1)

num_indices = []
cate_indices = []


for i in range(len(x.columns)):
    if(x.iloc[:,i].dtype == 'O'):
       cate_indices.append(i)
    
    elif(x.iloc[:,i].dtype == "int64" or x.iloc[:,i].dtype == "float64"):
        num_indices.append(i)


label_indices = []
onehot_indices = []

for i in cate_indices:
    if(len(x.iloc[:,i].value_counts()) == 2):
        label_indices.append(i)
        
    else:
        onehot_indices.append(i)
        
label  = LabelEncoder()


for i in label_indices:
    x.iloc[:,i] = label.fit_transform(x.iloc[:,i])         
        
ct = ColumnTransformer(transformers=[('encoder' , OneHotEncoder() , onehot_indices)] , remainder= 'passthrough')

x = ct.fit_transform(x)
x = x.toarray()


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x = sc.fit_transform(x)

from sklearn.model_selection import train_test_split
x_train  , x_test , y_train , y_test = train_test_split(x , y, test_size=0.2 ,random_state=0)


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=100)

regressor.fit(x_train , y_train)
y_pred = regressor.predict(x_test)



y_pred = y_pred.reshape(len(y_pred),1)
y_test = y_test.reshape(len(y_test),1)

concat = np.concatenate((y_pred,y_test),axis=1)