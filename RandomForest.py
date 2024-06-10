# -*- coding: utf-8 -*-
"""
Created on Thu May  2 14:18:02 2022

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

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10)
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)

#Sonuçlar;

print("------------------------------")
from sklearn.metrics import confusion_matrix
cm = confusion_matrix
rfc_cm = cm(y_test, y_pred)
print(cm)
print("------------------------------")
print("Random Forest Sonuçlar:")
print("------------------------------")

from sklearn.metrics import r2_score,accuracy_score,f1_score, precision_score
from sklearn import metrics
print("accuracy:")
print(accuracy_score(y_train,rfc.predict(X_train)))
print("------------------------------")

from sklearn.metrics import f1_score
print("f1 Score:")
print(f1_score(y_test, y_pred, average='macro'))
print("------------------------------")

from sklearn.metrics import precision_score
print("Precision:")
print(precision_score(y_test,y_pred, average='macro'))
print("------------------------------")