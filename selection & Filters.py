# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 14:57:41 2023

@author: ADMIN
"""
import pandas as pd
d={'Name':['abhinay','rushi','kartik','mahesh','gitesh'],
   'Exam':['sem1','sem2','sem1','sem2','sem2'],
   'Subject':['maths','science','maths','science','maths'],
   'Marks':[32,34,56,32,43]}

df=pd.DataFrame(d,columns=['Name','Exam','Subject','Marks'])
df
#select single column from data set.
df['Name']

# to select multiple columns from data frame
df[['Name','Subject','Marks']]

#select the row
df[1:2]

#to apply filter
df['Marks']>40
df[df['Marks']>40]


# select the pandas data frame
df.iloc[:2,2:5]
