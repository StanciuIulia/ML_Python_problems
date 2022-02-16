# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 17:25:53 2021

@author: iulia.stanciu
"""
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_squared_error

"""
citire si prelucrare date train
"""
data_labels_train = pd.read_csv('SPECT.train')
data_labels_train = data_labels_train.to_numpy()
nr_instances_train, nr_attributes_train = data_labels_train.shape

#print(data_labels_train)

data_train = data_labels_train[:,:-1]
labels_train = data_labels_train[:,-1]

print(data_train)
print(labels_train)
print(len(labels_train))
"""
citire si prelucrare date test
"""

data_labels_test = pd.read_csv('SPECT.test')
data_labels_test = data_labels_test.to_numpy()
nr_instances_test, nr_attributes_test = data_labels_test.shape

#print(data_labels_train)

data_test = data_labels_test[:,:-1]
labels_test = data_labels_test[:,-1]

print(data_test)
print(labels_test)
print(len(labels_test))


"""
SVM classifucation, linear kernel
"""
clf = svm.SVC(kernel='linear')
clf.fit(data_train,labels_train)
predictions = clf.predict(data_test)

accuracy_linear = accuracy_score(labels_test, predictions)
print('Acuratete pentru kernel liniar = ', accuracy_linear)

error_linear = mean_squared_error(labels_test, predictions)
print('Eroarea medie patratica pentru kernel liniar =', error_linear)

accuracy_linear = balanced_accuracy_score(labels_test, predictions)
print('Acuratete echilibrata pentru kernel liniar = ', accuracy_linear)

print()


"""
SVM classifucation, polynomial kernel
"""
clf = svm.SVC(kernel='poly', degree = 2)
clf.fit(data_train,labels_train)
predictions = clf.predict(data_test)


accuracy_poly = accuracy_score(labels_test, predictions)
print('Acuratete pentru kernel polinomial = ', accuracy_poly)

error_poly = mean_squared_error(labels_test, predictions)
print('Eroarea medie patratica pentru kernel polinomial =', error_poly)

accuracy_poly = balanced_accuracy_score(labels_test, predictions)
print('Acuratete echilibrata pentru kernel polinomial = ', accuracy_poly)
print()


"""
SVM classifucation, gaussian radial basis function kernel
"""
clf = svm.SVC(kernel='rbf',C=1,gamma=1)
clf.fit(data_train,labels_train)
predictions = clf.predict(data_test)


accuracy_rbf = accuracy_score(labels_test, predictions)
print('Acuratete pentru kernel gaussian radial= ', accuracy_rbf)

error_rbf = mean_squared_error(labels_test, predictions)
print('Eroarea medie patratica pentru kernel gaussian radial =', error_rbf)

accuracy_rbf = balanced_accuracy_score(labels_test, predictions)
print('Acuratete echilibrata pentru kernel gaussian radial= ', accuracy_rbf)
print()


"""
SVM classifucation, sigmoid kernel
"""
clf = svm.SVC(kernel='sigmoid')
clf.fit(data_train,labels_train)
predictions = clf.predict(data_test)


accuracy_sgm = accuracy_score(labels_test, predictions)
print('Acuratete pentru kernel sigmoid = ', accuracy_sgm)

error_sgm = mean_squared_error(labels_test, predictions)
print('Eroarea medie patratica pentru kernel sigmoid =', error_sgm)

accuracy_sgm = balanced_accuracy_score(labels_test, predictions)
print('Acuratete echilibrata pentru kernel sigmoid = ', accuracy_sgm)
print()