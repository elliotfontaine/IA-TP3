import load_datasets
import Classifiers # importer les classes du classifieur bayesien et Knn
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
TRAIN_RATIO = 0.8
IRIS_LABELS = ["setosa", "versicolor", "virginica"]
WINE_LABELS = [0,1]
ABALONE_LABELS = [0,1,2]



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
# print(abalone)


# Entrainez votre classifieur
iris_knn.train(iris['train'], iris['train_labels'])
wine_knn.train(wine['train'], wine['train_labels'])
abalone_knn.train(abalone['train'], abalone['train_labels'])

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
confusion_iris_knn = iris_knn.evaluate(iris['test'], iris['test_labels'])
confusion_wine_knn = wine_knn.evaluate(wine['test'], wine['test_labels'])
confusion_abalone_knn = abalone_knn.evaluate(abalone['test'], abalone['test_labels'])

print("IRIS DATASET (multi-classes)")
stat.print_stats(confusion_iris_knn)
""
print("WINE DATASET (binaire)")
stat.print_stats(confusion_wine_knn, binary=True)

print("ABALONE DATASET (multi-classes)")
stat.print_stats(confusion_abalone_knn)

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






