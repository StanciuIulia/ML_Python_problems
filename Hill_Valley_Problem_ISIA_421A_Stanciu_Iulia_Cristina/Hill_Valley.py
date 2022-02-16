import pandas as pd
import numpy as np
from sklearn import neural_network
from sklearn.metrics import accuracy_score

def read_file(file):
    date_etichete = pd.read_csv(file).to_numpy()

    return date_etichete[:,:-1], date_etichete[:,-1]


date_train1, etichete_train1 = read_file('Hill_Valley_without_noise_Training.data')
date_test1, etichete_test1 = read_file('Hill_Valley_without_noise_Testing.data')

date_train2, etichete_train2 = read_file('Hill_Valley_with_noise_Training.data')
date_test2, etichete_test2 = read_file('Hill_Valley_with_noise_Testing.data')

# instances_train_without_noise, attributes_train_without_noise = date_train1.shape
# instances_test_without_noise, attributes_test_without_noise = date_test1.shape
# print(instances_train_without_noise)
# print(instances_test_without_noise)

# instances_train_with_noise, attributes_train_with_noise = date_train2.shape
# instances_test_with_noise, attributes_test_with_noise = date_test2.shape
# print(instances_train_with_noise)
# print(instances_test_with_noise)

date_train_complet = date_train1 + date_train2
etichete_train_complet = etichete_train1 + etichete_train2
date_test_complet = date_test1 + date_test2
etichete_test_complet = etichete_test1 + etichete_test2


learning_rate = [0.01, 0.1]
nr_neuroni = 100
for i in learning_rate:
    clf = neural_network.MLPClassifier(hidden_layer_sizes=(nr_neuroni), learning_rate_init=i)

    clf.fit(date_train1, etichete_train1)
    predictii1 = clf.predict(date_test1)
    acratete_fara_zgomot = accuracy_score(etichete_test1, predictii1)
    print('Acuraterea pentru setul de date fara zgomot pentru 1 strat ascuns si learning rate de', i ,'=', acratete_fara_zgomot)
    
    clf.fit(date_train2, etichete_train2)
    predictii2 = clf.predict(date_test2)
    acuratete_cu_zgomot = accuracy_score(etichete_test2, predictii2)
    print('Acuraterea pentru setul de date cu zgomot pentru 1 strat ascuns si learning rate de', i ,'=', acuratete_cu_zgomot)
    
    clf.fit(date_train_complet, etichete_train_complet)
    predictii_complet = clf.predict(date_test_complet )
    acuratete_complet = accuracy_score(etichete_test_complet, predictii_complet)
    print('Acuraterea pentru setul de date intreg pentru 1 strat ascuns si learning rate de', i ,'=', acuratete_complet)
    print()
    
nr_neuroni_strat2 = [nr_neuroni, int(nr_neuroni/2)]
for i in learning_rate:
    for j in nr_neuroni_strat2:
        clf = neural_network.MLPClassifier(hidden_layer_sizes=(nr_neuroni, j), learning_rate_init=i)

        clf.fit(date_train1, etichete_train1)
        predictii1 = clf.predict(date_test1)
        acratete_fara_zgomot = accuracy_score(etichete_test1, predictii1)
        print('Acuraterea pentru setul de date fara zgomot pentru 2 straturi ascunse de', nr_neuroni, 'si', j,'neuroni si learning rate de', i ,'=', acratete_fara_zgomot)
    
        clf.fit(date_train2, etichete_train2)
        predictii2 = clf.predict(date_test2)
        acuratete_cu_zgomot = accuracy_score(etichete_test2, predictii2)
        print('Acuraterea pentru setul de date cu zgomot pentru 2 straturi ascunse de', nr_neuroni, 'si', j,'neuroni si learning rate de', i ,'=', acuratete_cu_zgomot)
        
        clf.fit(date_train_complet, etichete_train_complet)
        predictii_complet = clf.predict(date_test_complet )
        acuratete_complet = accuracy_score(etichete_test_complet, predictii_complet)
        print('Acuraterea pentru setul de date intreg pentru 2 straturi ascunse de', nr_neuroni, 'si', j,'neuroni si learning rate de', i ,'=', acuratete_complet)
        print()


