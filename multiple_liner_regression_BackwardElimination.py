# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 11:34:10 2023

@author: ADMIN
"""
# Multiple liner Regression

import pandas as pd

# import Dataset

dataset= pd.read_excel("F:\dataset_dummy_2.xlsx")
print(type(dataset))

# Method 1 (Handling Categocrical varibles)
pd.get_dummies(dataset['State'])
pd.get_dummies(dataset['State'],drop_first=True)
s_dummy=pd.get_dummies(dataset['State'],drop_first=True)
s_dummy.head(5)

# Now, lets concatenate these dummy var columns in our dataset.
dataset=pd.concat([dataset,s_dummy],axis=1)
dataset.head(5)

# droping column whose dummy var have created.
dataset.drop(['State',],axis=1,inplace=True)
dataset.head(5)

# Obtaining DV and IV from dataset.
x=dataset.iloc[:,[0,1,2,4,5]].values
y=dataset.iloc[:,-3].values

# Spliting the dataset for Training set and Testing set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

# Fitting multiple liner regression model into Training set.

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train, y_train)

# Predicting the test set results.
y_pred=regressor.predict(x_test)

#Accuracy of the model

from sklearn.metrics import r2_score
r2_score(y_test, y_pred)

#coefficient
regressor.coef_

#intercepts
regressor.intercept_

 #-------Backward Elimination--------------
 # Backward elimination is a feature selection tecnique while buildings a machine linear regression
 # to remove those fatures that do not have significants effect on dependent varibles.
 
 
 # step 1:- Preparation of backward elimination:
     
# importing the library:
import statsmodels.api as am

# Adding a column in matrix of features.
import numpy as nm
x=nm.append(arr=nm.ones((50,1)).astype(int), values=x,axis=1)

# Applying Backward elimination now.
#irstly we will create a new feature vector x_opt, which will only contain a set of 
# independent feature that are significantly affecting the dependent varible.

x_opt=x[:,[0,1,2,3,4,5]]

# for fitting the model, we will create a regressor_OLS object of new class OLS of
# stastsmodels librery. Then we will fit it by using the fit() method.

regrosser_OLS=am.OLS(endog= y, exog=x_opt).fit()

# we will use summary() method to get the summary table of all the varibles.
regrosser_OLS.summary()

# now since x5 has highest p-value grater than 0.05 hence, will remove the x5 varible 
# (dummy varibles) from the table and will refit the model.
x_opt=x[:,[0,1,2,3,4]]
regrosser_OLS=am.OLS(endog=y,exog=x_opt).fit()
regrosser_OLS.summary()

# now since x4 has highest p-value grater than 0.05 hence, will remove the x4 varible 
# (dummy varibles) from the table and will refit the model.
x_opt=x[:,[0,1,2,3]]
regrosser_OLS=am.OLS(endog=y, exog=x_opt).fit()
regrosser_OLS.summary()

# now we will remove x2 which is having 0.597 p-value
# and aging refit the model.

x_opt=x[:,[0,1,3]]
regrosser_OLS=am.OLS(endog=y, exog=x_opt).fit()
regrosser_OLS.summary()

# Finally, we will remove one more varible, which has .60 p-value for marketing spend,
# that is more then significant level value of 0.05.

x_opt=x[:,[0,1]]
regrosser_OLS=am.OLS(endog=y, exog=x_opt).fit()
regrosser_OLS.summary()


# Hence only R&D independent varible is a significant varible for the prediction.
# so we can now predict efficiently using this varible.

# Building Multile Regression model by only using R&D spend.

# importing dataset.
data_set=pd.read_excel("F:\dataset_dummy_2.xlsx")

# Extracting dependent and independent Varible.
x_BE=data_set.iloc[:,:-4].values
y_BE=data_set.iloc[:,4].values

# spliting dataset for training set & testing set.
from sklearn.model_selection import train_test_split
x_BE_train,x_BE_test,y_BE_train,y_BE_test=train_test_split(x_BE,y_BE,test_size=0.2,random_state=0)

# fitting the MLR model into traing set.
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_BE_train, y_BE_train)

# Predicting the test result set.
y_BE_pred=regressor.predict(x_BE_test)

# Checking the score/ Accuracy
# calculating the r2 value
from sklearn.metrics import r2_score
r2_score(y_BE_test, y_BE_pred)

# the above score tells that our model is now accurate with the test dataset with
# accuracy equal to 95%.


# calculating the coefficient.
print(regressor.coef_)

# calculating the intercept.
print(regressor.intercept_)

# Regression Equ. is

# profit=48422 + .85 * R&D_Spend.

# visualisation the result.
import matplotlib.pyplot as plt

line_chart1=plt.plot(x_BE_test, y_BE_pred, '--',c='red')
line_chart2=plt.plot(x_BE_test, y_BE_test,':',c='blue')

# using scatter plot
Department=data_set.iloc[:,0]
Profit=data_set.iloc[:,-1]

#col_array=['blue'] * 17 + ['green'] * 12 + ['red'] * 15

plt.scatter(Department, Profit, marker='*')
plt.xlabel('Department',fontsize=15)
plt.ylabel("Profit", fontsize= 16)

import seaborn as sns
sns.pairplot(data_set)
data_set.corr()
