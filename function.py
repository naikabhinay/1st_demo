# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 19:11:13 2023

@author: ADMIN
"""
# Absoulte function
interger=-20
abs(interger)

interger1=-90.034
abs(interger1)

# all function

k=[1,3,45,-4, ]
print(all(k))
# any 0 value in dataset that time it will show the false.
k1=[False, 0]
print(all(k1))


k3=[]
print(all(k3))

# bool functions.

test1=[0]
print(bool(test1))

test2=[1]
print(bool(test2),)

test3=[]
print(bool(test3))

list1=[1,3,5,5]
sum(list1)
print(sum(list1))
sum(list1,10)
print(sum(list1))

len(list1)

#list create in python
Gaurav= list()
string='abcde'
print(list(string))

# divmod used to give quotient and reminder of two number
result=divmod(10, 2)
print(result)

# dict its a constructore which create a dictionary.
resullt1= dict()
resullt1

resullt2= dict(a=1,b=2)
print(resullt2)


#set is used for new set but not include duplicate valur.
par=set()
print(par)
par2=set('12')
print(par2)
par3=set('javatpoint')
print(par3)
par4={1,2,2}
print(par4)

part5= set('javatpoint')
print(part5)

print(pow(2, 3))

print(pow(-3, 3))


#tuple() used to tuple object
t1= tuple()
print(t1)


li=[1,3,4,5,]
t2=tuple(li)
print(t2)


t1=tuple('java')
t1


#lambda function

x=lambda a ,b:a+b
x
print(x(12,12))


# program to filter out the list which contain number divisible by 3.

lis=[1,3,5,7,9,23,45,23,56]
addlist=list(filter(lambda x:(x%3==0), lis))
print(addlist)


#program to triple each number of the list using map.
l1=[1,2,3,4,5,6,7,8,33,22,12]
triple=list(map(lambda x:(x*3),l1))
print(triple)

