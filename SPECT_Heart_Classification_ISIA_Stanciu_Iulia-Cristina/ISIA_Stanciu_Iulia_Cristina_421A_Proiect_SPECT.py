"""
Created on Wed Nov 17 17:25:53 2021

@author: iulia.stanciu
"""

import math
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier



""" functie citire fisier si despartire date in atribute si etichete """
def read_data(file):
    data_labels = pd.read_csv(file, header=None).to_numpy()
    
    #functia verifica daca exista valori lipsa marcate cu "?" si sterge instanta cu valori lipsa
    data_labels = np.array([row for row in data_labels if '?' not in row])
    #nr_instances, nr_attributes = data_labels.shape
    
    # functia returneaza datele si etichetele
    return data_labels[:,:-1], data_labels[:,-1]



""" functie verificare corelatie intre atribute """
def delete_corelated_data(d_train, d_test):
    for col1 in range(np.shape(d_train)[1]-1, 0, -1):
        for col2 in range(col1-1, -1, -1):
            check_eq = np.equal(d_train[:, col1], d_train[:, col2])
            check_not_eq = np.not_equal(d_train[:, col1], d_train[:, col2])

            if(np.all(check_eq) or np.all(check_not_eq)):
                d_train = np.delete(d_train, col2, 1)
                d_test = np.delete(d_test, col2, 1)
                break
    # functia returneaza datele de antrenare si cele de testare dupa ce se scapa de atributele nefolositoare        
    return d_train, d_test


data_train, labels_train = read_data('SPECT.train')
data_test, labels_test = read_data('SPECT.test')

data_train, data_test = delete_corelated_data(data_train, data_test)

nr_instances_train, nr_attributes_train = data_train.shape
#print(nr_instances_train, nr_attributes_train)

nr_instances_test, nr_attributes_test = data_test.shape
#print(nr_instances_test, nr_attributes_test)

# KNN prediction

accuracy_knn = []

balanced_accuracy_knn=[]


for i in range(1, (int)(math.sqrt(nr_instances_train)) ):
    knn = KNeighborsClassifier(n_neighbors = i, metric='euclidean')
    knn.fit(data_train,labels_train)
    predictions = knn.predict(data_test)
    
    accuracy_knn.append(accuracy_score(labels_test, predictions))
    #print(accuracy_score(labels_test, predictions))
    balanced_accuracy_knn.append(balanced_accuracy_score(labels_test, predictions))
    #print(balanced_accuracy_score(labels_test, predictions))
    print()
    
print('Acuratete maxima pentru knn =', max(accuracy_knn))
print('Acuratetea echilibrata maxima pentru knn =', max(balanced_accuracy_knn))
print('Acestea se obtin pentru',  accuracy_knn.index(max(accuracy_knn))+1,'vecini.')
print()


# SVM classifucation, linear kernel

accuracy_linear = []
balanced_accuracy_linear=[]

for i in range(-5, 8, 2):
    clf = svm.SVC(kernel='linear', C=2**i)
    clf.fit(data_train,labels_train)
    predictions = clf.predict(data_test)

    accuracy_linear.append(accuracy_score(labels_test, predictions))
    #print('Acuratete pentru kernel liniar pentru C=',2**i , ' = ', accuracy_score(labels_test, predictions))

    balanced_accuracy_linear.append(balanced_accuracy_score(labels_test, predictions))
    #print('Acuratete echilibrata pentru kernel liniar pentru C=',2**i , ' = ', balanced_accuracy_score(labels_test, predictions))

    #print()

print('Acuratetea maxima pentru kernel liniar =', max(accuracy_linear))
print('Acuratetea echilibrata maxima pentru kernel liniar =', max(balanced_accuracy_linear))
print('Acestea se obtin pentru parametrul Cost =',  2**((accuracy_linear.index(max(accuracy_linear))*2-5)))
print()



#SVM classifucation, polynomial kernel

accuracy_poly = []
balanced_accuracy_poly=[]

for grad in range(2,8):
    for i in range(-5, 8, 1):
        clf = svm.SVC(kernel='poly', degree = grad, C=2**i)
        clf.fit(data_train,labels_train)
        predictions = clf.predict(data_test)
    

        accuracy_poly.append(accuracy_score(labels_test, predictions))
        #print('Acuratete pentru kernel polinomial = ', accuracy_score(labels_test, predictions))

        balanced_accuracy_poly.append(balanced_accuracy_score(labels_test, predictions))
        #print('Acuratete echilibrata pentru kernel polinomial = ', balanced_accuracy_score(labels_test, predictions))
        #print()
        
print('Acuratetea maxima pentru kernel polinomial =', max(accuracy_poly))
print('Acuratetea echilibrata maxima pentru kernel polinomial =', max(balanced_accuracy_poly))

print()

"""
#SVM classifucation, gaussian radial basis function kernel

clf = svm.SVC(kernel='rbf',C=1,gamma=1)
clf.fit(data_train,labels_train)
predictions = clf.predict(data_test)


accuracy_rbf = accuracy_score(labels_test, predictions)
print('Acuratete pentru kernel gaussian radial= ', accuracy_rbf)

accuracy_rbf = balanced_accuracy_score(labels_test, predictions)
print('Acuratete echilibrata pentru kernel gaussian radial= ', accuracy_rbf)
print()


#SVM classifucation, sigmoid kernel

clf = svm.SVC(kernel='sigmoid')
clf.fit(data_train,labels_train)
predictions = clf.predict(data_test)


accuracy_sgm = accuracy_score(labels_test, predictions)
print('Acuratete pentru kernel sigmoid = ', accuracy_sgm)

accuracy_sgm = balanced_accuracy_score(labels_test, predictions)
print('Acuratete echilibrata pentru kernel sigmoid = ', accuracy_sgm)
print()
"""