
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
imputer = SimpleImputer(missing_values=np.nan,strategy='mean',verbose=0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

#Encoding catagorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],  
    remainder='passthrough')                                  
X= np.array(ct.fit_transform(X),dtype=np.int64)

labelencoder_y= LabelEncoder()
Y = labelencoder_y.fit_transform(Y.ravel())

#Splitting the dataset into training set and Test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,
                                test_size=0.2,random_state=0)

#feature scaling (scale the features of test to all same level)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train)
x_test = sc_X.transform(x_test)






