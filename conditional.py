# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 15:15:58 2023

@author: ADMIN
"""
# program for even or od number
num=int(input("Enter the number" ))
if num%2==0:
        print(num,"is even")
else:
     print(num,"is odd number")
     
     
 # program for grater or smaller using elseif
     
a=int(input("enter the 1st num:- "))
b=int(input("enter the 2nd num:- "))
c=int(input("enter the 3rd num:- "))

if a>b and a>c :
    print(a,'a element is largest')
elif  b>c and b>a:
    print(b,"b element is largest")
elif c>a and c>b :
    print(c,"c element is largest")
elif a==b or b==c or c==a:
    print("number is repeated")
else:
    print("no number found")
# person elisible for voating or not

age=int(input("Enter Your Age:-"))
if age>=18:
    print("your age is ", age ,"& you Are eligible for voting")
else:
    print("your age", age ,"is not eligible for voting. sorry!!")
    
    
# for loop

i=0

for i in range(0,10):
    print(i, end=",")
    
    
# printing the table of given number.
i=1
num=int(input("Enter the number:-"))
for i in range(1,11):
   
   
    
print("%a x %a = %a" %(num, i, num*i))
