"""
par la suite vous pourrez valider votre implémentation en comparant les résultats obtenus avec ceux obtenus en utilisant les méthodes de scikit-learn.
"""

#Ecris moi du code pour classifier les données du Iris Dataset en utilisant le modèle des k plus proches voisins (KNN) avec k=5 et distance euclidienne.
#Utilise la library scikit-learn.

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import *
from load_datasets import *

# Charger les données d'entraînement et de test
TRAIN_RATIO = 0.8

# Charger/lire les datasets
keys = ['train', 'train_labels', 'test', 'test_labels']
iris = dict(zip(keys, load_iris_dataset(TRAIN_RATIO)))
wine = dict(zip(keys, load_wine_dataset(TRAIN_RATIO)))
abalone = dict(zip(keys, load_abalone_dataset(TRAIN_RATIO)))

# Instancier le classificateur KNN avec le nombre de voisins souhaité
k = 5 # Exemple de nombre de voisins = 5
knn_iris = KNeighborsClassifier(n_neighbors=k)
knn_wine = KNeighborsClassifier(n_neighbors=k)
knn_abalone = KNeighborsClassifier(n_neighbors=k)

# Entraîner le modèle sur les données d'entraînement
knn_iris.fit(iris['train'], iris['train_labels'])
knn_wine.fit(wine['train'], wine['train_labels'])
knn_abalone.fit(abalone['train'], abalone['train_labels'])

# Faire des prédictions sur les données de test
iris_label_pred = knn_iris.predict(iris['test'])
wine_label_pred = knn_wine.predict(wine['test'])
abalone_label_pred = knn_abalone.predict(abalone['test'])

# Calculer les métriques
for dataset, y_pred in [(iris, iris_label_pred), (wine, wine_label_pred), (abalone, abalone_label_pred)]:
    y_test = dataset['test_labels']
    # Calculer la matrice de confusion
    confusion = confusion_matrix(y_test, y_pred)
    print("\nMatrice de confusion :")
    print(confusion)

    # Calculer l'exactitude du modèle
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy :   ", accuracy)

    # Calculer la précision du modèle
    precision = precision_score(y_test, y_pred, average='weighted')
    print("Précision :  ", precision)
    
    # Calculer le rappel (recall) du modèle
    recall = recall_score(y_test, y_pred, average='weighted')
    print("Rappel :     ", recall)
    
    # Calculer le F1-score du modèle
    f1 = f1_score(y_test, y_pred, average='weighted')
    print("F1-score :   ", f1)
