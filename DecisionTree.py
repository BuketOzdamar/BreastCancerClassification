# -*- coding: utf-8 -*-
"""
Created on Sat May 2 13:53:52 2022

@author: BUKET
"""

import pandas as pd
import numpy as np
from sklearn.tree import plot_tree

#Data oku;
data = pd.read_csv('data.csv')
#print(data)

#Bağımlı-bağımsız değişken belirle;
X = data.drop(["Id", "Malignite" ],axis=1)
#print(X)
y = data["Malignite"]
#print(y)

#Train-test kümesi oluşturma;
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20, random_state=0)

#model inşaası;
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)
y_pred = dt.predict(X_test)

#Sonuçlar;
print("------------------------------")
from sklearn.metrics import confusion_matrix
cm = confusion_matrix
dt_cm = cm(y_test, y_pred)
print(cm)
print("------------------------------")
print("Decision Trees Sonuçlar:")
print("------------------------------")
from sklearn.metrics import r2_score,accuracy_score,f1_score, precision_score
from sklearn import metrics
print("accuracy:")
print(accuracy_score(y_train,dt.predict(X_train)))
print("------------------------------")

from sklearn.metrics import f1_score
print("f1 Score:")
print(f1_score(y_test, y_pred, average='macro'))
print("------------------------------")

from sklearn.metrics import precision_score
print("Precision:")
print(precision_score(y_test,y_pred, average='macro'))
print("------------------------------")

