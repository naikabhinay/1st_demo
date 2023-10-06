# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 11:32:53 2023

@author: ADMIN
"""

import matplotlib.pyplot as plt
import numpy as np

city=['delhi','kokan','pune','jablpur','kota']
happiness_index=[34,56,23,55,78]
plt.bar(city, happiness_index, color='yellow',edgecolor='red')
plt.xlabel('city', fontsize=16)
plt.ylabel('happiness_index', fontsize=16)
plt.title('Barchart-Happiness_Index', fontsize=20)

# horizontal Bar Chart

city=['delhi','kokan','pune','jablpur','kota']
happiness_index=[34,56,23,55,78]
plt.barh(city, happiness_index, color='yellow',edgecolor='red')
plt.xlabel('city', fontsize=16)
plt.ylabel('happiness_index', fontsize=16)
plt.title('Barchart-Happiness_Index', fontsize=20)

#stacked bar chart in python with legends.
city=['delhi','kokan','pune','jablpur','kota']
gender=['Male','Female']
Happiness_Index_Male=[60,56,45,76,98]
Happiness_Index_Female=[30,44,55,32,56]
plt.bar(city, Happiness_Index_Male,color='pink',edgecolor='black')
plt.bar(city, Happiness_Index_Female,color='green',edgecolor='black')
plt.xlabel('city',fontsize=16)
plt.ylabel('happiness_index', fontsize=16)
plt.title('Stacked_Bar_Chart_Happiness_index',fontsize=18)
plt.legend(gender,loc=2)

# Histogram with no Fills.
values=[44,54,67,87,56,45,34,23,87,64,58,98,42,78,90]
plt.hist(values,5, color='cyan',align='mid',histtype='step',edgecolor='black',label='Test score data')
plt.legend(loc=2)

# Histogram withFills.
values=[44,54,67,87,56,45,34,23,87,64,58,98,42,78,90]
plt.hist(values,5, color='cyan',align='mid',histtype='bar',edgecolor='black',label='Test score data')
plt.legend(loc=2)

#Box Plot

Value1=[2,56,32,56,12,76,98,67,15,34,54,87,90]
value2=[22,65,76,90,94,67,2,87,80,60,50,89,94]
value3=[55,95,46,60,96,57,75,87,90,20,40,1,44]
value4=[56,45,16,20,16,50,5,27,50,60,10,32,44]

box_plot_value_data=[Value1,value2,value3,value4]
plt.boxplot(box_plot_value_data)

#Box Plot fills and label

Value1=[2,56,32,56,12,76,98,67,15,34,54,87,90]
value2=[22,65,76,90,94,67,2,87,80,60,50,89,94]
value3=[55,95,46,60,96,57,75,87,90,20,40,1,44]
value4=[56,45,16,20,16,50,5,27,50,60,10,32,44]

box_plot_value_data=[Value1,value2,value3,value4]
plt.boxplot(box_plot_value_data,patch_artist=True,labels=['cource1','cource2','cource3','cource4'])

#Horizontal Box Plot fills and label

Value1=[2,56,32,56,12,76,98,67,15,34,54,87,90]
value2=[22,65,76,90,94,67,2,87,80,60,50,89,94]
value3=[55,95,46,60,96,57,75,87,90,20,40,1,44]
value4=[56,45,16,20,16,50,5,27,50,60,10,32,44]

box_plot_value_data=[Value1,value2,value3,value4]
box=plt.boxplot(box_plot_value_data,vert=0,patch_artist=True,labels=['cource1','cource2','cource3','cource4'])

# Line Plot or Line Chart
v1=[2,4,6,8,4,5,9,8]
plt.plot(v1, color='red')

# Multiple line with legands and lables.

v2=[2,4,6,4,9,1,6]
v3=[4,7,4,9,1,5,3]
plt.plot(v2,color='pink')
line2=plt.plot(v3,color='green')
plt.xlabel('Range 1')
plt.ylabel('Range 2')
plt.title('ratio')

#pie chart
values= [60,80,34,10,32]
col=['b','g','r','p','y']
labels=['uk','us','india','china','korea']
exp=(0.5,0,0,0,0)
plt.pie(values,labels=labels,colors=col,explode=exp,autopct='%1.1f%%',shadow=True)


# Scatter plot
Weight1=[63.5,54.6,76.5,78.5,98.5,45.7,87.8,65.8,65.8]
Height1=[145.5,155.7,156,189.2,127.7,146.8,156.7,122.9,155.9]

Weight2=[68.5,74.6,86.5,48.5,68.5,35.7,97.8,45.8,85.8]
Height2=[195.5,125.7,176,149.2,147.7,136.8,196.7,142.9,155.9]

Weight3=[58.5,54.6,56.5,88.5,48.5,55.7,67.8,75.8,65.8]
Height3=[125.5,195.7,166,169.2,127.7,176.8,156.7,162.9,155.9]

import numpy as np

Height=np.concatenate((Height1, Height2, Height3))
Weight=np.concatenate((Weight1,Weight2  ,Weight3))

#col_array=['blue'] * 17 + ['green'] * 12 + ['red'] * 15

plt.scatter(Weight, Height, marker='*')
plt.xlabel('weight',fontsize=15)
plt.ylabel("height", fontsize= 16)
