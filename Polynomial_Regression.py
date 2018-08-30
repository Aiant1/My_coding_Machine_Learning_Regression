# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 12:47:21 2018

@author: Antika
"""

#POLYNOMIAL REGRESSION

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Read the csv file
dataset = pd.read_csv('C:\\Users\\ASUS\\Desktop\\Linear_algebra\\Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:,2].values

#fitting the linear regression
from sklearn.linear_model import LinearRegression
regression_lin=LinearRegression() 
regression_lin.fit(x,y)

#fitting polynominal regression

from sklearn.preprocessing import PolynomialFeatures
regression_poly = PolynomialFeatures(degree=2)
x_poly=regression_poly.fit_transform(x)
regression_lin2=LinearRegression() 
regression_lin2.fit(x_poly,y)
#visualize polynomial regression

plt.scatter(x,y,color ="orange")
plt.plot(x,regression_lin2.predict(x_poly))
plt.title("linear_regression")
plt.xlabel("position")
plt.ylabel("salary")


from sklearn.preprocessing import PolynomialFeatures
regression_poly = PolynomialFeatures(degree=3)
x_poly=regression_poly.fit_transform(x)
regression_lin2=LinearRegression() 
regression_lin2.fit(x_poly,y)
#visualize polynomial regression

plt.scatter(x,y,color ="k")
plt.plot(x,regression_lin2.predict(x_poly))
plt.title("linear_regression")
plt.xlabel("position")
plt.ylabel("salary")


from sklearn.preprocessing import PolynomialFeatures
regression_poly = PolynomialFeatures(degree=5)
x_poly=regression_poly.fit_transform(x)
regression_lin2=LinearRegression() 
regression_lin2.fit(x_poly,y)
#visualize polynomial regression

plt.scatter(x,y,color ="y")
plt.plot(x,regression_lin2.predict(x_poly))
plt.title("linear_regression")
plt.xlabel("position")
plt.ylabel("salary")


#visualize linear regression
plt.scatter(x,y,color ="r")
plt.plot(x,regression_lin.predict(x))
plt.title("linear_regression")
plt.xlabel("position")
plt.ylabel("salary")






