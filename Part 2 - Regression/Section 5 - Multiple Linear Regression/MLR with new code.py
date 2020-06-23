# -*- coding: utf-8 -*-
"""
Created on Mon May 25 11:45:21 2020

@author: LOKESH
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Importing the dataset
# dataset = pd.read_csv('50_Startups.csv')
dataset = pd.read_csv('D:\\Machine-Learning-A-Z-New\\Machine Learning A-Z New\\Part 2 - Regression\\Section 5 - Multiple Linear Regression\\50_Startups.csv')

dataset.head()
# visualize the data using pairplot

sns.pairplot(dataset)

# splitting the data
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

print(X,'\n')
print(y)

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])


transformer = ColumnTransformer(
    transformers=[
        ("OneHot",        # Just a name
         OneHotEncoder(), # The transformer class
         [3]              # The column(s) to be applied on.
         )
    ],
    remainder='passthrough' # donot apply anything to the remaining columns
)

X = transformer.fit_transform(X.tolist())
X = X.astype('float64')

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

print(X)
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# Predicting the Test set results
y_pred = regressor.predict(X_test)


# INterpretting coefficents

print(regressor.intercept_)
print(regressor.coef_)

# error metrics

from sklearn.metrics import mean_absolute_error, mean_squared_error,accuracy_score

print(mean_absolute_error(y_test, y_pred))
print(mean_squared_error(y_test,y_pred))
print(np.sqrt(mean_squared_error(y_test,y_pred)))

# accuracy score - round off all the values 
Y_test = [np.round(value) for value in y_test]
predictions = [np.round(value) for value in y_pred]

# accuracy test

print(accuracy_score(Y_test,predictions))





