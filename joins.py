# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 11:45:16 2023

@author: ADMIN
"""

import pandas as pd

#data frame 1
d1={'customer_id':pd.Series([1,2,3,4,5]),
    'product_id':pd.Series(['oven','oven','oven',
                           'tv','light'])}

pd.DataFrame(d1)


#data frame 2

d2={'customer_id':pd.Series([1,2,3,5,6]),
    'name':pd.Series(['abhinay','naik','rushi',
                           'sakshi','galgate'])}

pd.DataFrame(d2)

# inner join()
#return only rows where left table have matching keys in the right table

print(pd.merge(d1, d2, on='customer_id',how='inner'))
      
print(pd.merge(d1, d2, on='customer_id',how='outer'))


import pandas as pd

# Data frame 1
d1 = {'customer_id': [1, 2, 3, 4, 5],
      'product_id': ['oven', 'oven', 'oven', 'tv', 'light']}
df1 = pd.DataFrame(d1)

# Data frame 2
d2 = {'customer_id': [1, 2, 3, 5, 6],
      'name': ['abhinay', 'naik', 'rushi', 'sakshi', 'galgate']}
df2 = pd.DataFrame(d2)

# Inner join: Returns only rows where both DataFrames have matching customer_id
print("Inner Join:")
print(pd.merge(df1, df2, on='customer_id', how='inner'))



# Outer join: Returns all rows from both DataFrames, filling in missing values with NaN
print("\nOuter Join:")
print(pd.merge(df1, df2, on='customer_id', how='outer'))



# Right Join: return all rows from right column.

print(pd.merge(df1, df2, on="customer_id", how= 'right')) 


#left join : return all rows from left table.


print(pd.merge(df1, df2, on="customer_id", how='left'))


