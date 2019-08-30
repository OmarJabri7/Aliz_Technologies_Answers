# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 15:28:35 2019

@author: omarj
"""

import pandas as pd
from Threshold import ThresholdBinarizer
from Estimator import custom_editor

def main():
    #Main computations:
    # Importing the dataset
    dataset = pd.read_csv('Social_Network_Ads.csv')
    X = dataset.iloc[:, [2, 3]].values
    y = dataset.iloc[:, 4].values
    
    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
    
    # Feature Scaling to avoid high variance between the attributes
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    #importing the Logistic regression model to use in the transformer and binarizer
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(random_state = 0)
    
    #fitting the model in the binarizer
    tb = ThresholdBinarizer(lr)
    
    tb.fit(X_train,y_train)
    
    T = tb.findThreshold(X_train)
    
    X = tb.transform(X_test)
    
    classifier = custom_editor()
    
    classifier.fit(X,y_train)
    
    y_pred = classifier.predict(X,T)
    
    #finding the conufsion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    #finding the accuracy score of the model based on the confusion matrix
    from sklearn.metrics import accuracy_score
    score = accuracy_score(y_pred,y_test)
    
    print(score)
    
    
if __name__ == "__main__":
    main()