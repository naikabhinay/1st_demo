# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 11:11:46 2023

@author: ADMIN
"""

import pandas as pd
import numpy as np


# create a data frame
df1={'subject':['sem1','sem2','sem3','sem4','sem5','sem6'],
    'score':[45,76,np.nan,65,np.nan,87]}
df1=pd.DataFrame(df1,columns=['subject','score'])
df1


# to check any is null value in data frame.
df1.isnull()
df1.notnull()

# to check any null value in column

df1.isnull().any()

# check how many missing value across column

df1.isnull().sum()

# Droping missing value with rows.

#creating a Data Frame

df2={'Name':['Abhinay','kartik','rushi','pranay','rushab',np.nan,np.nan]
     ,'State':['maharashtra','karnataka','rajasthan','banglore',np.nan,'jammu',np.nan]
     ,'Gender':['male','male',np.nan,'male',np.nan,'male',np.nan]
     ,'Age':[45,67,87,66,98,np.nan,np.nan]}
df2=pd.DataFrame(df2,columns=['Name','State','Gender','Age'])
df2
 
# to drop all the rows having null values.
df2.dropna()

#drop having all the records are null or nan

df2.dropna(how='all')

# drop only if a row has more than 2 nan values
df2.dropna(thresh=2)

# drop nan in a specific column

df2.dropna(subset='Gender')
df2.dropna(subset=['Gender','Age'])

# droping row using axis value
df2.dropna(axis=0)

## droping column using axis value
 
df2.dropna(axis=1)


#creating a Data Frame again.

df3={'Name':['Abhinay','kartik','rushi','pranay','rushab',np.nan,np.nan]
     ,'State':['maharashtra','karnataka','rajasthan','banglore',np.nan,'jammu',np.nan]
     ,'Gender':['male','male',np.nan,'male',np.nan,'male',np.nan]
     ,'Age':[45,67,87,66,98,np.nan,np.nan]}
df3=pd.DataFrame(df3,columns=['Name','State','Gender','Age'])
df3

# Replace null value with 0
df3.fillna(0) 

#Replace replace null or missing value with mean or median.

df3['Age'].fillna(df3['Age'].mean(),inplace=True)
df3

df3['Age'].fillna(df3['Age'].median(),inplace=True)
df3

df4={'subject':[12,13,14,15,16,1000],
     'marks':[45,65,54,76,0,5000],
     'passing':[45,65,5000,56,76,87]}

df4=pd.DataFrame(df4,columns=['subject','marks','passing'])
# replace missing value with specific value
df4
df4.replace({5000:50,1000:10})


# handling duplicate value

df4={'name':['abhinay','kartik','tanmay','abhinay'],
     'marks':[12,23,43,12]}
df4

info=pd.DataFrame(df4)
info

info=info.drop_duplicates()
print(info)

df5={'name':['abhinay','kartik','tanmay','abhinay'],
     'marks':[12,23,43,21]}
df5

info=pd.DataFrame(df5)
info

info=info.drop_duplicates()
print(info)