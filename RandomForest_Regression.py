# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 11:00:05 2018

@author: Antika

"""

# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 09:38:39 2018

@author: Antika

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Read the csv file
dataset = pd.read_csv('C:\\Users\\ASUS\Desktop\\Linear_algebra\\Position_Salaries.csv')

x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values
 


#fitting non linear no_continuous regression model(RANDOM_FOREST_REGRESSION) 
from sklearn.ensemble import RandomForestRegressor
regression = RandomForestRegressor() 
regression.fit(x,y)

#predicting what shoud be the salary of the employee at position level 6.5

y_predict = regression.predict(6.5)


#visualize the grapgh in higher resolution & smooth

x_grid = np.arange(min(x),max(x),0.10)
x_grid = x_grid.reshape(len(x_grid),1)
plt.scatter(x,y,color ="red")
plt.plot(x_grid,regression.predict(x_grid))
plt.title("ex vs sal")
plt.xlabel("experience ")
plt.ylabel("salary")

