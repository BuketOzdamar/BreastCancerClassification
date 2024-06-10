# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 22:25:01 2022

@author: BUKET
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

#Data oku;
data = pd.read_csv('data.csv')
#print(data)

#Bağımlı-bağımsız değişken belirle;
X = data.drop(["Id", "Malignite" ],axis=1)
print(X)
y = data["Malignite"]
print(y)

#Train-test kümesi oluşturma;
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20, random_state=0)

from sklearn.svm import SVC
svc = SVC (kernel='rbf')
svc.fit(X_train,y_train)

y_pred= svc.predict(X_test)

#Sonuçlar;
from sklearn.metrics import confusion_matrix
cm = confusion_matrix
svc_cm = cm(y_test, y_pred)
print(cm)
print("------------------------------")
print("Support Vector Machine Sonuçlar:")

from sklearn.metrics import accuracy_score
print("------------------------------")
print("Accuracy degeri:")
print(accuracy_score(y_train,svc.predict(X_train)))

from sklearn.metrics import f1_score
print("------------------------------")
print("f1 Score:")
print(f1_score(y_test, y_pred, average='macro'))

from sklearn.metrics import precision_score
print("------------------------------")
print("Precision:")
print(precision_score(y_test,y_pred, average='macro'))































