import pandas as pd
import numpy as np
from sklearn import svm
#from sklearn.metrics import accuracy_score, balanced_accuracy_score


# functie pentru citirea datelor din fisier
def read_data(file, colnames):
    # citeste datele din fisier avand numeel coloanelor specificate
    data_labels = pd.read_csv(file, names=colnames, header=None, low_memory=False)
    
    # renunta la primul rand din matrice (acesta continea numele coloanelor)
    data_labels.drop(0, inplace=True)
    
    # renunta la coloanele cu atribute non-predictive: id_1 si id_2 
    # si la coloanele xu prea multe valori lipsa: cmp_fname_c2 si cmp_lname_c2
    data_labels.drop(['id_1', 'id_2', 'cmp_fname_c2', 'cmp_lname_c2'], axis=1, inplace=True)

    # functia returneaza datele si etichetele
    return data_labels

col_names = ['id_1', 'id_2', 'cmp_fname_c1', 'cmp_fname_c2', 'cmp_lname_c1', 'cmp_lname_c2',
             'cmp_sex', 'cmp_bd', 'cmp_bm', 'cmp_by', 'cmp_plz', 'is_match']

filenames = ['block_1.csv', 'block_2.csv', 'block_3.csv', 'block_4.csv', 'block_5.csv',
             'block_6.csv', 'block_7.csv', 'block_8.csv', 'block_9.csv', 'block_10.csv']

dataframes = []
for file in filenames:
    dataframes.append(read_data(file, col_names))
    
data_and_labels = []   
# concateneaza datele din cele 10 fisiere
data_and_labels = pd.concat(dataframes).to_numpy()

# sterge randurile cu valori lipsa notate cu '?'
data = np.array([row for row in data_and_labels if '?' not in row])

# afiseaza dimensiunile setului de date procesat: randuri=instante 
# si coloane=atribute si etichete
print(data.shape)

# setul de date este impartit in procente 
# 25% din date fac parte din setul de test, iar restul de 75% din cel de train
data_test, data_train = np.split(data, [int(.25*len(data))])

# atributele setului de train
X_train = data_train[:,:-1]
# etichetele setului de train
Y_train = data_train[:,-1]

# atributele setului de test
X_test = data_test[:,:-1]
# etichetele setului de test
Y_test = data_test[:,-1]

C=[0.03125,0.125,0.5,2,8,32,128]
for i in range(len(C)):
    clf = svm.SVC(kernel='linear',C=C[i]).fit(X_train, Y_train)
    pred=clf.predict(X_test)
    cont=0
    for j in range(len(Y_test)):
        if Y_test[j]==pred[j]:
            cont+=1
    print('Acuratetea este '+str(cont/len(Y_test))+' pentru parametrul cost = '+ str(C[i]))