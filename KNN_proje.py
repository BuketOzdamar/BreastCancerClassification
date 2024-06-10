# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 14:25:28 2022

@author: BUKET
"""
#Kütüphaneler;
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Data oku;
data = pd.read_csv('data.csv')
print(data)

#Bağımlı-bağımsız değişken belirle;
X = data.drop(["Id", "Malignite" ],axis=1)
print(X)
y = data["Malignite"]
print(y)

#Train-test kümesi oluşturma;
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20, random_state=0)

#Obje oluşturup eğitime geç;
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)

#Sonuçlar;
print("------------------------------")
from sklearn.metrics import confusion_matrix
cm = confusion_matrix
knn_cm = cm(y_test, y_pred)
print(cm)
print("------------------------------")
print("KNN Sonuçlar:")

from sklearn.metrics import accuracy_score
print("------------------------------")
print("KNN Accuracy degeri:")
print(accuracy_score(y_train,knn.predict(X_train)))

print("------------------------------")
from sklearn.metrics import f1_score
print("f1 Score:")
print(f1_score(y_test, y_pred, average='macro'))

print("------------------------------")
from sklearn.metrics import precision_score
print("Precision:")
print(precision_score(y_test,y_pred, average='macro'))
print("------------------------------")

#cor_mat = np.corrcoef(X.T)
#print ("Correlation matrisinin sekli:", cor_mat.shape)



















































