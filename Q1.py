# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 19:29:25 2018

@author: uqrvaism
"""

from sklearn.datasets import make_blobs, make_friedman1
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import neighbors


X_train = pd.read_csv("C:\Users\YifeiWang\Downloads\Prac4\Xtrain.csv").as_matrix()
y_train = pd.read_csv("C:\Users\YifeiWang\Downloads\Prac4\Ytrain.csv").as_matrix()
X_validate = pd.read_csv("C:\Users\YifeiWang\Downloads\Prac4\Xvalidate.csv").as_matrix()
y_validate = pd.read_csv("C:\Users\YifeiWang\Downloads\Prac4\Yvalidate.csv").as_matrix()
X_test = pd.read_csv("C:\Users\YifeiWang\Downloads\Prac4\Xtest.csv").as_matrix()
y_test = pd.read_csv("C:\Users\YifeiWang\Downloads\Prac4\Ytest.csv").as_matrix()



n_arr = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,12,22,23,24,25,26,27,28,29,30,
         31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50]

scores = list()
for k in n_arr:
    knn = neighbors.KNeighborsRegressor(k)
    knn.fit(X_train,y_train)
    y_val_pred = knn.predict(X_validate)
    err = np.mean(np.power( y_val_pred - y_validate,2))
    scores.append(err)
    print(k, " - ", err)
    
plt.plot(n_arr, scores)
plt.ylabel('CV score (mean squared error)')
plt.xlabel('k - # of neighbours')
plt.axhline(np.max(scores), linestyle='--', color='.5')

# find the best
kstar_val = min(scores) 
k_star = np.argmin(scores)

print("best k is ",k_star, " validation mse = ", kstar_val)

# find the generalization error
knn = neighbors.KNeighborsRegressor(k_star)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
err = np.mean(np.power( y_pred - y_test,2))

print("The generalization losss is  ", err)

####################################################
# this is why we need 3 sets!
####################################################

