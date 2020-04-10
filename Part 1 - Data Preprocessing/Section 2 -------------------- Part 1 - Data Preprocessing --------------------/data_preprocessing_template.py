# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 00:56:58 2020

@author: abhisek
"""

#Importing libraries 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing dataset 
datasets = pd.read_csv('Data.csv')
X = datasets.iloc[:,:-1].values 
Y = datasets.iloc[:,3:].values 

#Taking care of missing data 
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

#Encoding catagorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
print("X",X)
#onehotencoder = OneHotEncoder(categories='auto')
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],  
    remainder='passthrough')                                  
X= ct.fit_transform(X)




