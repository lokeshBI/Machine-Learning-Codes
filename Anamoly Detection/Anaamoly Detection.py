
# import packages 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# loading data
dataset = pd.read_csv('D:\\Machine-Learning-A-Z-New\\Machine Learning A-Z New\\Part 3 - Classification\\Section 14 - Logistic Regression\\Social_Network_Ads.csv')

dataset.head()

# drop unnecessary columns 
dataset.drop(columns = 'User ID', inplace = True)

dataset.head()

# get dummies for all variables
df = pd.get_dummies(dataset)
df.head()

# scale the data
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler() 
df2 = scaler.fit_transform(df)
df2 = pd.DataFrame(df2, columns = df.columns)

df2.head()

########## algorithms for anamoly detection
###### DBSCAN
###### Isolation forrest....and few

#### DBSCAN
from sklearn.cluster import DBSCAN
outlier_detection = DBSCAN(eps = .2, metric='euclidean', min_samples = 5,n_jobs = -1)
clusters = outlier_detection.fit_predict(df2)

clusters # it will create an array of numbers which -1 represents anamoly

unique_elements, counts_elements = np.unique(clusters, return_counts=True)
print(np.asarray((unique_elements, counts_elements)))

from matplotlib import cm
cmap = cm.get_cmap('Set1')
df2.plot.scatter(x='EstimatedSalary',y='Age', c=clusters, cmap=cmap,
 colorbar = False)
plt.show()

## isolation forest

from sklearn.ensemble import IsolationForest
rs=np.random.RandomState(0)
clf = IsolationForest(max_samples=100,random_state=rs, contamination=.1) 
clf.fit(df2)
if_scores = clf.decision_function(df2)
if_anomalies=clf.predict(df2)
if_anomalies=pd.Series(if_anomalies).replace([-1,1],[1,0])
if_anomalies=df2[if_anomalies==1]

df2.head()

cmap=np.array(['white','red'])
plt.scatter(df2.iloc[:,1],df2.iloc[:,2],c='white',s=20,edgecolor='k')
plt.scatter(if_anomalies.iloc[:,0],if_anomalies.iloc[:,1],c='red')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.title('Isolation Forests - Anomalies')
plt.show()












