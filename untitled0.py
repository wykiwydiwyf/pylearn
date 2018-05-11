# -*- coding: utf-8 -*-
"""
Created on Wed May  9 13:46:55 2018

@author: YifeiWang
"""

from sklearn.datasets import make_blobs, make_friedman1
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


credit = pd.read_csv(r"C:\Users\Yifei\Downloads\Prac4\Prac4\crx.data.csv", na_values=['?'])

credit.dtypes



# drop rows with missing values
credit = credit.dropna()


lb_make = LabelEncoder()
credit['A1'] = lb_make.fit_transform(credit['A1'])
credit['A4'] = lb_make.fit_transform(credit['A4'])
credit['A5'] = lb_make.fit_transform(credit['A5'])
credit['A6'] = lb_make.fit_transform(credit['A6'])
credit['A9'] = lb_make.fit_transform(credit['A9'])
credit['A10'] = lb_make.fit_transform(credit['A10'])
credit['A12'] = lb_make.fit_transform(credit['A12'])
credit['A13'] = lb_make.fit_transform(credit['A13'])
credit['A16'] = lb_make.fit_transform(credit['A16'])




def crossvali(X,y,model):

    classifiers = [
    ("KNeighbors", KNeighborsClassifier()),
    ("Perceptron", Perceptron()),
    ("Random Forest", RandomForestClassifier()),
    ("SVMs", svm.SVC()),
    ("SAG", LogisticRegression())
]

    scores_list = None
    
    for name, clf in classifiers:
        print("training %s" % name)
        clf.fit(X, y)
        scores = cross_val_score(clf, X, y)
        print("scores %s" % scores.mean())
        scores_list.append(scores)
    
    
    
crossvali(credit[['A2'],['A3']],credit['A16'],model=1)