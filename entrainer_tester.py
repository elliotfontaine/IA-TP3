import numpy as np
import sys
import load_datasets
import Classifiers # importer les classes du classifieur bayesien et Knn
#importer d'autres fichiers et classes si vous en avez développés
import statistiques as stat

"""
C'est le fichier main duquel nous allons tout lancer
Vous allez dire en commentaire c'est quoi les paramètres que vous avez utilisés
En gros, vous allez :
1- Initialiser votre classifieur avec ses paramètres
2- Charger les datasets
3- Entrainer votre classifieur
4- Le tester

"""

# Initialisez vos paramètres
TRAIN_RATIO = 0.8
IRIS_LABELS = [0,1,2]
WINE_LABELS = [0,1]



# Initialisez/instanciez vos classifieurs avec leurs paramètres
iris_knn = Classifiers.Knn(k=5)
wine_knn = Classifiers.Knn(k=5)
abalone_knn = Classifiers.Knn(k=5)
#iris_bayes = Classifiers.NaiveBayes()
#wine_bayes = Classifiers.NaiveBayes()
#abalone_bayes = Classifiers.NaiveBayes()


# Charger/lire les datasets
keys = ['train', 'train_labels', 'test', 'test_labels']
iris = dict(zip(keys, load_datasets.load_iris_dataset(TRAIN_RATIO)))
wine = dict(zip(keys, load_datasets.load_wine_dataset(TRAIN_RATIO)))
abalone = dict(zip(keys, load_datasets.load_abalone_dataset(TRAIN_RATIO)))


# Entrainez votre classifieur
iris_knn.train(iris['train'], iris['train_labels'])
wine_knn.train(wine['train'], wine['train_labels'])

"""
Après avoir fait l'entrainement, évaluez votre modèle sur 
les données d'entrainement.
IMPORTANT : 
    Vous devez afficher ici avec la commande print() de python,
    - la matrice de confusion (confusion matrix)
    - l'accuracy
    - la précision (precision)
    - le rappel (recall)
    - le F1-score
"""




# Tester votre classifieur

print("\nK-nearest neighbors avec k=5 et distance euclidienne \n")
confusion_iris_knn = iris_knn.evaluate(iris['test'], iris['test_labels'], labels=IRIS_LABELS, distance_type='euclidean')
confusion_wine = wine_knn.evaluate(wine['test'], wine['test_labels'], labels=WINE_LABELS, distance_type='euclidean')

print("IRIS DATASET (multi-classes)")
print("Matrice de confusion: ", confusion_iris_knn)
print("Exactitude:  ", stat.accuracy(confusion_iris_knn))
print("Macro-Precision: ", stat.macro_precision(confusion_iris_knn))
print("Macro-Rappel:    ", stat.macro_recall(confusion_iris_knn))
print("Macro-F1-score:  ", stat.macro_f1_score(confusion_iris_knn))
print("Weighted-Precision: ", stat.weighted_precision(confusion_iris_knn))
print("Weighted-Rappel:    ", stat.weighted_recall(confusion_iris_knn))
print("Weighted-F1-score:  ", stat.weighted_f1_score(confusion_iris_knn), "\n")

print("WINE DATASET (binaire)")
print("Matrice de confusion: ", confusion_wine)
print("Exactitude:  ", stat.accuracy(confusion_wine))
print("Precision:   ", stat.class_precision(confusion_wine, 1))
print("Rappel:      ", stat.class_recall(confusion_wine, 1))
print("F1-score:    ", stat.class_f1_score(confusion_wine, 1), "\n")

"""
Finalement, évaluez votre modèle sur les données de test.
IMPORTANT : 
    Vous devez afficher ici avec la commande print() de python,
    - la matrice de confusion (confusion matrix)
    - l'accuracy
    - la précision (precision)
    - le rappel (recall)
    - le F1-score
"""






