# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 15:42:42 2023

@author: ADMIN
"""
# import librery
import pandas as pd
import matplotlib.pyplot as plt

# import dataset 
dataset=pd.read_csv("E:\python practice dataset\Mall_Customers.csv")
dataset.head(5)
x=dataset.iloc[:,[3,4]].values

# using the elbow method to find the optimal number of clusters.
from sklearn.cluster import KMeans
help(KMeans())
import os
import warnings
from sklearn.cluster import KMeans

os.environ['OMP_NUM_THREADS'] = '1'
# Ignore the warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0, n_init=10)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('No of clusters')
plt.ylabel('WCSS')

# Fitting k-means to the dataset

kmeans=KMeans(n_clusters=5,init='k-means++',random_state=42)
y_kmeans=kmeans.fit_predict(x)

kmeans=pd.DataFrame(y_kmeans)
dataset1=pd.concat([dataset,kmeans],axis=1)

# visualising the clusters.
plt.scatter(x[y_kmeans==0,0],x[y_kmeans==0,1], s=100, c='pink',label='cluster 1')
plt.scatter(x[y_kmeans==1,0],x[y_kmeans==1,1], s=100, c='red',label='cluster 2')
plt.scatter(x[y_kmeans==2,0],x[y_kmeans==2,1], s=100, c='yellow', label='cluster 3')
plt.scatter(x[y_kmeans==3,0],x[y_kmeans==3,1], s=100, c='green',label='cluster 4')
plt.scatter(x[y_kmeans==4,0],x[y_kmeans==4,1], s=100, c='blue', label='cluster 5')
plt.scatter(x[y_kmeans==5,0],x[y_kmeans==5,1], s=100, c='magenta',label='cluster 6')
plt.scatter(x[y_kmeans==6,0],x[y_kmeans==6,1], s=100, c='magenta',label='cluster 7')
plt.scatter(x[y_kmeans==7,0],x[y_kmeans==7,1], s=100, c='magenta',label='cluster 8')

plt.title('clusters of customer')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score(1-100)')
plt.legend()
plt.show()
