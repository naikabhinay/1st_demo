# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 15:36:55 2023

@author: ADMIN
"""

import pandas as pd

# import dataset
dataset=pd.read_csv("E:\python practice dataset\Purchase_data.csv")

#Creating Dummy varible for fields
sex_dummy=pd.get_dummies(dataset['Gender'],drop_first=True)
sex_dummy.head(5) 

# Concaneate the dummy var to dataset.

dataset=pd.concat([dataset,sex_dummy],axis=1) 

# Drop field which is dummy variable created.
dataset.drop(["Gender"],axis=1,inplace=True)

x= dataset.iloc[:,[1,2,3]].values
y= dataset.iloc[:,4]

# Spliting the dataset for training ad testing set.
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)

# Fitting random forest model to dataset
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=70,criterion='entropy',max_depth=3,min_samples_leaf=5)
classifier.fit(x_train, y_train)

# to see len of the decision tree.
len(classifier.estimators_)

# to access the perticular decision tree use indexing

classifier.estimators_[0]

# Predicting test set result.
y_pred=classifier.predict(x_test)

# Making consfusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)
cm
classifier.feature_importances_
# Accuracy= 95%
# Accuracy for without gender feild is= 46,50,55,60%
# using change in n_estimators.

# Decision tree -1 visualization
from sklearn import tree
import matplotlib.pyplot as plt
fig,axes=plt.subplots(nrows=1,ncols=1,figsize=(4,4), dpi=1000)
cn=['0','1']
tree.plot_tree(classifier.estimators_[0],class_names=cn,filled=True)

# Decision tree 2- visualization
from sklearn import tree
import matplotlib.pyplot as plt
fig,axes=plt.subplots(nrows=1,ncols=1,figsize=(4,4),dpi=1000)
cn=['0','1']
tree.plot_tree(classifier.estimators_[1],class_names=cn,filled=True)

