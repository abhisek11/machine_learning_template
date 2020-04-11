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

#Splitting the dataset into training set and Test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,
                                test_size=0.2,random_state=0)

#feature scaling (scale the features of test to all same level)
'''from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train)
x_test = sc_X.transform(x_test)'''






