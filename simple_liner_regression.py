# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 15:18:48 2023

@author: ADMIN
"""
# simple linear regression

# Import Libraries.

import pandas as pd
import matplotlib.pyplot as plt

# Importing data from dataset

dataset=pd.read_excel("F:\\dataset_dummy_1.xlsx")
print(type(dataset))

x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values


# splitting the dataeet into the training set and test set

from sklearn.model_selection import train_test_split
x_train,x_test, y_train, y_test = train_test_split(x,y, test_size= 1/3, random_state=0)

# Note :- The parameter 'random_state'  is used to randomly bifurcate the dataset into training &
# testing dataset. Tht number should be supplied as arguments to parameter 'random_state'
# which helps us get the max accuracy. and that number is decided by hit & trail method.

# fitting simple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train, y_train)

# calculating the coefficients.

print(regressor.coef_)


# calculating the intercepts.
print(regressor.intercept_)

# predicting the test set result 
y_pred= regressor.predict(x_test)

# accuracy of the model

# calculating the r square values.

from sklearn.metrics import r2_score
r2_score(y_test, y_pred)

# create the DataFrame

df1= {'Actual_Applicants':y_test,
      'Predicted_Applicants':y_pred}
df1=pd.DataFrame(df1, columns=['Actual_Applicants','Predicted_Applicants'])
print(df1)

# visualising the predicted results.

line_chart1=plt.plot(x_test, y_pred, '--',c='red')
line_chart2=plt.plot(x_test, y_test,':',c='blue')