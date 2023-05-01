import load_datasets
import Classifiers # importer les classes du classifieur bayesien et Knn
import numpy as np
#importer d'autres fichiers et classes si vous en avez développés
import statistiques as stat # importer les fonctions de calcul des métriques

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
TRAIN_RATIO = 0.7



# Initialisez/instanciez vos classifieurs avec leurs paramètres
#iris_knn = Classifiers.Knn(k=5)
#wine_knn = Classifiers.Knn(k=5)
#abalone_knn = Classifiers.Knn(k=5)
iris_arbre = Classifiers.DecisionTree()
wine_arbre = Classifiers.DecisionTree()
abalone_arbre = Classifiers.DecisionTree()


# Charger/lire les datasets
keys = ['train', 'train_labels', 'test', 'test_labels']
iris = dict(zip(keys, load_datasets.load_iris_dataset(TRAIN_RATIO)))
wine = dict(zip(keys, load_datasets.load_wine_dataset(TRAIN_RATIO)))
abalone = dict(zip(keys, load_datasets.load_abalone_dataset(TRAIN_RATIO)))
# print(abalone)


# Entrainez votre classifieur
#iris_knn.train(iris['train'], iris['train_labels'])
#wine_knn.train(wine['train'], wine['train_labels'])
#abalone_knn.train(abalone['train'], abalone['train_labels'])
"""
iris_accuracy = []
for i in range(len(iris['train'])):
    data_temp = []
    labels_temp = []
    for j in range(0, i + 1):
        data_temp.append(iris['train'][j])
        labels_temp.append(iris['train'][j])

    data_subset = np.array(data_temp)
    labels_subset = np.array(labels_temp)
    iris_arbre.train(data_subset, labels_subset, list(range(0,len((iris['train'])[0]))), 30, 100000000, 20000000000)
    confusion_iris_arbre = iris_arbre.evaluate(iris['test'], iris['test_labels'])
    iris_accuracy.append(stat.accuracy(confusion_iris_arbre))
print(iris_accuracy)    
""" 
    

iris_arbre.train(iris['train'], iris['train_labels'], list(range(0,len((iris['train'])[0]))), 30, 100000000, 20000000000)
wine_arbre.train(wine['train'], wine['train_labels'], list(range(0,len((wine['train'])[0]))), 30, 100000000, 20000000000)
#abalone_arbre.train(abalone['train'], abalone['train_labels'], list(range(0,len((abalone['train'])[0]))), 30, 100000000, 20000000000)

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
# confusion_iris_knn = iris_knn.evaluate(iris['train'], iris['train_labels'])
# confusion_wine_knn = wine_knn.evaluate(wine['train'], wine['train_labels'])
# confusion_abalone_knn = abalone_knn.evaluate(abalone['train'], abalone['train_labels'])

# print("IRIS DATASET (multi-classes)")
# stat.print_stats(confusion_iris_knn)
# ""
# print("WINE DATASET (binaire)")
# stat.print_stats(confusion_wine_knn, binary=True)

# print("ABALONE DATASET (multi-classes)")
# stat.print_stats(confusion_abalone_knn)



# Tester votre classifieur

print("\n█████████████ K-nearest neighbors avec k=5 et distance euclidienne █████████████\n")
#confusion_iris_knn = iris_knn.evaluate(iris['test'], iris['test_labels'])
#confusion_wine_knn = wine_knn.evaluate(wine['test'], wine['test_labels'])
#confusion_abalone_knn = abalone_knn.evaluate(abalone['test'], abalone['test_labels'])
confusion_iris_arbre = iris_arbre.evaluate(iris['test'], iris['test_labels'])
confusion_wine_arbre = iris_arbre.evaluate(wine['test'], wine['test_labels'])
#confusion_abalone_arbre = iris_arbre.evaluate(iris['test'], iris['test_labels'])


print("IRIS DATASET (multi-classes)")
#stat.print_stats(confusion_iris_knn)
stat.print_stats(confusion_iris_arbre)
""
print("WINE DATASET (binaire)")
#stat.print_stats(confusion_wine_knn, binary=True)
stat.print_stats(confusion_wine_arbre)

print("ABALONE DATASET (multi-classes)")
#stat.print_stats(confusion_abalone_knn)
#stat.print_stats(confusion_abalone_arbre)

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






