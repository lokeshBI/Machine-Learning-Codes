

# Ensemble Learning

# import packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# loading data
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv(url, names=names)
data.head()
data.shape
data.tail()

# splitting data

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# split the data into train and test


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

X_train.shape
X_test.shape

# k fold cross validation
kfolds = KFold(n_splits=10, random_state= 0)

# create the sub models
estimators = []
model1 = LogisticRegression()
estimators.append(('logistic', model1))
model2 = DecisionTreeClassifier()
estimators.append(('cart', model2))
model3 = SVC()
estimators.append(('svm', model3))

estimators

ensemble = VotingClassifier(estimators)
results = cross_val_score(ensemble, X_train, y_train, cv=kfolds)
print(results.mean())

modelfit = ensemble.fit(X_train, y_train)

y_pred = modelfit.predict(X_test)
y_pred


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

accuracy_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)
cm

print(classification_report(y_test, y_pred))










