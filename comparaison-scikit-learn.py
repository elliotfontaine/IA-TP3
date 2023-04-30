"""
par la suite vous pourrez valider votre implémentation en comparant les résultats obtenus avec ceux obtenus en utilisant les méthodes de scikit-learn.
"""

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import *
from load_datasets import *

# Charger les données d'entraînement et de test
TRAIN_RATIO = 0.7
HIDDEN_LAYERS = (10)

# Charger/lire les datasets
keys = ['train', 'train_labels', 'test', 'test_labels']
iris = dict(zip(keys, load_iris_dataset(TRAIN_RATIO)))
wine = dict(zip(keys, load_wine_dataset(TRAIN_RATIO)))
abalone = dict(zip(keys, load_abalone_dataset(TRAIN_RATIO)))

# Instancier le réseau de neurones
mlp_iris = MLPClassifier(hidden_layer_sizes=HIDDEN_LAYERS, max_iter=1000)
mlp_wine = MLPClassifier(hidden_layer_sizes=HIDDEN_LAYERS, max_iter=1000)
mlp_abalone = MLPClassifier(hidden_layer_sizes=HIDDEN_LAYERS, max_iter=1000)

# Entraîner le modèle sur les données d'entraînement
mlp_iris.fit(iris['train'], iris['train_labels'])
mlp_wine.fit(wine['train'], wine['train_labels'])
mlp_abalone.fit(abalone['train'], abalone['train_labels'])

# Faire des prédictions sur les données de test
iris_label_pred = mlp_iris.predict(iris['test'])
wine_label_pred = mlp_wine.predict(wine['test'])
abalone_label_pred = mlp_abalone.predict(abalone['test'])

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

