# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 12:09:48 2023

@author: ADMIN
"""

# Logestic Regression

# Import Librery
import pandas as pd
import seaborn as sns

# import data

titanic_data=pd.read_csv("F:/titanic.csv")
print(type(titanic_data))
titanic_data.head(5)
titanic_data.tail(5)

print("No of passengers in original dataset:-",str(len(titanic_data.index)))

# Analysing Data

sns.countplot(x='Survived',data=titanic_data)
sns.countplot(x='Survived',hue='Sex',data=titanic_data)
sns.countplot(x='Survived',hue='Pclass',data=titanic_data)

# Checking data type of a varible and converting it into another type....

titanic_data.info()
titanic_data["Age"].plot.hist()
titanic_data["Fare"].plot.hist()

# Identifying/ Finding missing values if any----
titanic_data.isnull()
titanic_data.isnull().sum()

# Note:
# Since missing values in "Embarked" are quite less, we cann deleted such rows.
# Since missing value in "Age" are high, its better we do imputation in it.


sns.heatmap(titanic_data.isnull(),yticklabels=False,cmap='viridis')   
sns.boxenplot(x='Age',data=titanic_data)

titanic_data.dropna(subset=['Embarked'],inplace=True)
sns.heatmap(titanic_data.isnull(),yticklabels=False,cmap='viridis')

# by box plot we observe that the no. of outliers in "age" are quite less, hence,
# if we plan to do imputation in "age" we can do it by "mean" imputation.

# Hndling Missing Value.

titanic_data.head(5)

#Imputation missing values in columns (Age)  with mean imputation.
titanic_data["Age"].fillna(titanic_data["Age"].mean(),inplace=True)
sns.heatmap(titanic_data.isnull(),yticklabels=False)

#Hence, we do not have any missing values in the dataset now.
titanic_data.isnull().sum()

# there are lot of string value var in dataet which have to be converted to numerical
# value for applying machine learning alorithms. Hence, we will now convert string var
#to numerical var.

titanic_data.info()
pd.get_dummies(titanic_data["Sex"])
Sex_dummy=pd.get_dummies(titanic_data["Sex"],drop_first=True)

titanic_data.info()
pd.get_dummies(titanic_data["Embarked"])
Embarked_dummy=pd.get_dummies(titanic_data["Embarked"],drop_first=True)

titanic_data.info()
pd.get_dummies(titanic_data["Pclass"])
Pclass_dummy=pd.get_dummies(titanic_data["Pclass"],drop_first=True)

# Now, lets concatenate the dummy var colums in our dataset.
titanic_data=pd.concat([titanic_data,Sex_dummy,Embarked_dummy,Pclass_dummy],axis=1)
titanic_data.head(5)

#Droping this columns who has dummy varible created.
titanic_data.drop(["PassengerId","Sex","Embarked","Pclass","Name","Ticket"],axis=1,inplace=True)
titanic_data.head(5)

# Splitting the dataset into Training And testing set.
x=titanic_data.drop(["Survived"],axis=1)
y=titanic_data["Survived"]
x
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.25,random_state=0)

# Convert column names to strings 
x_train.columns = x_train.columns.astype(str)
x_test.columns = x_test.columns.astype(str)

from sklearn.linear_model import LogisticRegression

logmodel=LogisticRegression(solver='liblinear')
logmodel.fit(x_train, y_train)

predictors=logmodel.predict(x_test)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, predictors)

# Hence, Accuracy =(112+58)/(112+20+33+58)=76.23%
# Calculting the coefficient.
print(logmodel.coef_)
# Calculating The intercepts.
print(logmodel.intercept_)

# to improve the accuracy of the model, lets go with Backward Elimination method &
# Rebuild the logistic model agin with few independent varibles.....

titanic_data_1=titanic_data
titanic_data_1.head(5)

# BAckward Elimination.

# Backward elimintion is a feature selection tecnique while building a machine learning model. It is tecnique
# to remove those feature that do not have siginificant effect on dependent varible or prediction of model.

# step 1:- Prepration of backward elimination
# import the Librery.
import statsmodels.api as sm

# Adding the column in matrix of features:
x1=titanic_data_1.drop(['Survived'],axis=1)
y1=titanic_data_1['Survived']

import numpy as np
x1=np.append(arr=np.ones((889,1)).astype(int), values=x1,axis=1)

# Applying Backward elimination process

# firstly we create the x_opt feature vector, which will only contain a set of
# independent feaures that are significantly affecting the dependent varibles.

x_opt=x1[:,[0,1,2,3,4,5,6,7,8,9]]

# for fitting the model, we will create a regressor_ols object of new ols of statsmodel librery
# then we will fit it by using the fit() method.

regressor_ols=sm.OLS(endog=y1,exog=x_opt).fit()
regressor_ols.summary()

# in the above summary table, we can clearly see the p-value of all the varible.
# and remove the ind var with p-value grater than 0.05.
x_opt=x1[:,[0,1,2,3,4,5,7,8,9]]
regressor_ols=sm.OLS(endog = y1,exog=x_opt).fit()
regressor_ols.summary()

x_opt=x1[:,[0,1,2,4,5,7,8,9]]
regressor_ols=sm.OLS(endog=y1,exog=x_opt).fit()
regressor_ols.summary()

x_opt=x1[:,[0,1,2,5,7,8,9]]
regressor_ols=sm.OLS(endog=y1,exog=x_opt).fit()
regressor_ols.summary()

# Hence, independent varible- Age, Sibsp, Sex, Pclass & Embarked varible
# for the predicting the value of Dependent var "Survived"

# so we can predict efficiently using these varibles.

# building logistic regression.

from sklearn.model_selection import train_test_split
x_BE_train,x_BE_test,y_BE_train,y_BE_test=train_test_split(x_opt,y1,test_size=0.25,random_state=0)


# fitting logistic regression to the train set.
from sklearn.linear_model import LogisticRegression
logmodel1=LogisticRegression(solver='liblinear')
logmodel1.fit(x_BE_train,y_BE_train)

predictors1=logmodel1.predict(x_BE_test)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_BE_test,predictors1)  

# Accuracy = (110+60)/(110+22+31+60)= 77%

print(logmodel1.intercept_)

print(logmodel1.coef_)

