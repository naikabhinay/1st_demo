# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 10:34:42 2023

@author: ADMIN
"""
# Multiple Linear regression

import pandas as pd
import seaborn as cd


# importing data from datase (excel file)

dataset=pd.read_excel("F:/dataset_dummy.xlsx")
print(type(dataset))
print(dataset.head())
# data visualization
cd.heatmap(dataset)

x=dataset.iloc[:,1:].values
y=dataset.iloc[:,0].values

# spliting data set into training set and testing set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y, test_size=0.3, random_state=1)
#plt.scatter()

# Note :- The parameter 'random_state'  is used to randomly bifurcate the dataset into training &
# testing dataset. Tht number should be supplied as arguments to parameter 'random_state'
# which helps us get the max accuracy. and that number is decided by hit & trail method.

# Fitting Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(x_train, y_train)

# Calculating the coefficients:
print(regressor.coef_)

# Calculating the intercepts:
print(regressor.intercept_)

# Predicting the test set model.
y_pred=regressor.predict(x_test)



# Accuracy of the model.

# Calculating the r squared value.

from sklearn.metrics import r2_score
r2_score(y_test, y_pred)
 
#Creating a dataframe
df1={'Actual Application':y_test,
                 'Predict Application':y_pred}
df1=pd.DataFrame(df1,columns=['Actual Application','Predict Application'])
print(df1)

# Visualization the prdicted results.
import matplotlib.pyplot as plt
line_chart1=plt.plot(y_pred, x_test,'--',c='green')
line_chart2=plt.plot(y_test,x_test,':',c='red')





















