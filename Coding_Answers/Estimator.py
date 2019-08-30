# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 17:00:29 2019

@author: omarj
"""
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
#this class will perform predictions based on the threshold extracted 
#from the previous class combined with the built-in logistic regression model
class custom_editor(BaseEstimator, ClassifierMixin):
    #fit the model to the new X testing set
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self
    
    def predict(self, X, T):
        # the implementation used here breaks ties differently
        # from the one used in Logistic Regressions:
        #return self.classes_.take(np.argmax(X, axis=1), axis=0)
        return np.where(X[:, 0]>T, *self.classes_)
    
    