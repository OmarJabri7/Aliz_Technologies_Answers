# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 16:56:20 2019

@author: omarj
"""
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
#this class will highlight the bet threshold based on gini impurity metric
class ThresholdBinarizer(BaseEstimator, TransformerMixin):
    #Setting the classifier as looogistic regression
    def __init__(self, clf):
        self.clf = clf
    #fitting the classifier to the data sets that we have
    def fit(self, X, y):
        self.clf.fit(X, y)
        return self
    #Gini impurity method to optimize threshold and then 
    #average out the results to find the optimal threshold
    def gini(p):
        return 1 - np.square(p)
    #finding the threshold from the training set
    def findThreshold(self, X):
        p = self.clf.predict_proba(X)
        p = p[:,0]
        T = np.average(p)
        return T
    #trsnforming the test set into a set of probabilities that 
    #will be compared wwith the threshold
    def transform(self, X):
        return self.clf.predict_proba(X)