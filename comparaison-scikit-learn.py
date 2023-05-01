"""
Recherche d'hyperparamètres
"""

from sklearn.neural_network import MLPClassifier
from load_datasets import *
from sklearn.metrics import *

def k_folds(X, y, k):
    """
    Divise un ensemble de données en k folds pour la validation croisée.
    X: tableau np.array 2D d'exemples (n_examples, n_features)
    y: tableau np.array 1D d'étiquettes (n_examples,)
    k: nombre de folds
    Retourne une liste de tuples (train, train_labels, test, test_labels) pour chaque fold.
    Un fold est une partition de l'ensemble de données complet, pas juste un k-ième de l'ensemble.
    """
    n_examples = len(X)
    fold_size = n_examples // k 
    remaining_examples = n_examples % k
    folds = []
    start = 0
    for fold in range(k):
        end = start + fold_size
        if fold < remaining_examples: # répartir les restes sur les premiers folds
            end += 1
        test_indexes = range(start, end)
        train_indexes = list(set(range(n_examples)) - set(test_indexes)) # différence d'ensembles
        train = X[train_indexes]
        train_labels = y[train_indexes]
        validation = X[test_indexes]
        validation_labels = y[test_indexes]
        folds.append((train, train_labels, validation, validation_labels))
        start = end
    return folds


TRAIN_RATIO = 0.7
N_NEURONS = (2, 4, 6, 8, 10, 15, 20, 25) # valeurs à tester pour le nombre de neurones dans la couche cachée
N_LAYERS = (1, 2, 3, 4, 5) # valeurs à tester pour nombre de couches cachées
MAX_ITER = 1000


# Charger/lire les datasets
keys = ['train', 'train_labels', 'test', 'test_labels']
iris = dict(zip(keys, load_iris_dataset(TRAIN_RATIO)))
wine = dict(zip(keys, load_wine_dataset(TRAIN_RATIO)))
abalone = dict(zip(keys, load_abalone_dataset(TRAIN_RATIO)))

# Création des listes de 5 folds pour la validation croisée
# Chaque fold est un tuple (train, train_labels, test, test_labels)
iris_folds = k_folds(iris['train'], iris['train_labels'], 5)
wine_folds = k_folds(wine['train'], wine['train_labels'], 5)
abalone_folds = k_folds(abalone['train'], abalone['train_labels'], 5)

""" # Choix du nombre de neurones dans la couche cachée
for dataset in [iris_folds, wine_folds, abalone_folds]:
    mean_accuracies = []
    for weidth in N_NEURONS:
        mlp = MLPClassifier(hidden_layer_sizes=(weidth), max_iter=MAX_ITER)
        accuracy = []
        for fold in dataset:
            mlp.fit(fold[0], fold[1])
            y_pred = mlp.predict(fold[2])
            accuracy.append(np.mean(fold[3] == y_pred)) # calculer l'exactitude
        mean_accuracies.append(np.mean(accuracy))
    print(mean_accuracies) """

# => Le nombre de neurones optimal est 10 pour les 3 datasets
OPTIM_N = 10

""" # Choix du nombre de couches cachées
for dataset in [iris_folds, wine_folds, abalone_folds]:
    mean_accuracies = []
    for depth in N_LAYERS:
        mlp = MLPClassifier(hidden_layer_sizes=(OPTIM_N,)*depth, max_iter=MAX_ITER)
        accuracy = []
        for fold in dataset:
            mlp.fit(fold[0], fold[1])
            y_pred = mlp.predict(fold[2])
            accuracy.append(np.mean(fold[3] == y_pred)) # calculer l'exactitude
        mean_accuracies.append(np.mean(accuracy))
    print(mean_accuracies) """

# => Le nombre de couches cachées optimal est 3 pour les 3 datasets
OPTIM_DEPTH = 3

# Instancier le réseau de neurones
mlp_iris = MLPClassifier(hidden_layer_sizes=(OPTIM_N,)*OPTIM_DEPTH, max_iter=1000)
mlp_wine = MLPClassifier(hidden_layer_sizes=(OPTIM_N,)*OPTIM_DEPTH, max_iter=1000)
mlp_abalone = MLPClassifier(hidden_layer_sizes=(OPTIM_N,)*OPTIM_DEPTH, max_iter=1000)

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

