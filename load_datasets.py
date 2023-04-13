import numpy as np
import random


def _load_dataset(filename, train_ratio, lenght_data):
  """
  Cette fonction privée a pour but de lire un dataset. Elle est appelée par les fonctions load_iris_dataset,
  load_wine_dataset, et load_abalone_dataset.

  Args:
      filename: le chemin vers le fichier du dataset.
      train_ratio: le ratio des exemples (ou instances) qui vont servir pour l'entrainement,
      le reste des exemples va etre utilisé pour les test.
      lenght_data: la nombre de colonnes dans le dataset. On suppose que les labels sont dans la dernière colonne.
      
  Retours:
        Cette fonction doit retourner 4 matrices de type Numpy: train, train_labels, test, et test_labels

        - train : une matrice numpy qui contient les exemples qui vont etre utilisés pour l'entrainement, chaque 
        ligne dans cette matrice représente un exemple d'entrainement.

        - train_labels : contient les labels (ou les étiquettes) pour chaque exemple dans train, de telle sorte
          que : train_labels[i] est l'etiquette pour l'exemple train[i]

        - test : une matrice numpy qui contient les exemples qui vont etre utilisés pour le test, chaque 
        ligne dans cette matrice représente un exemple de test.

        - test_labels : contient les étiquettes pour chaque exemple dans test, de telle sorte
          que : test_labels[i] est l'etiquette pour l'exemple test[i]
  """
  
  random.seed(1) # Pour avoir les meme nombres aléatoires à chaque initialisation.
  
  # Le fichier du dataset est dans le dossier datasets en attaché 
  f = open(filename, 'r')
  
  train_ratio = float(train_ratio)
  if train_ratio < 0 or train_ratio > 1:
    raise ValueError('Le ratio doit être compris entre 0 et 1')
  
  # Lecture des exemples et de leurs étiquettes.
  examples = []
  labels = []
  for line in f:
      data = line.strip().split(',')
      if len(data) == lenght_data:
          try: # Pour les datasets qui ont des valeurs numériques.
            example = [float(data[i]) for i in range(lenght_data - 1)]
          except ValueError: # Pour les datasets qui ont certaines valeurs catégorielles.
            example = [data[i] for i in range(lenght_data - 1)]
          label = data[lenght_data - 1]
          examples.append(example)
          labels.append(label)
      else:
          raise IOError('Le fichier du dataset est corrompu')
  f.close()

  # Mélange des exemples.
  data = list(zip(examples, labels))
  random.shuffle(data)
  examples, labels = zip(*data)

  # Calcul du nombre d'exemples à utiliser pour l'entrainement.
  n_train_examples = int(len(examples) * train_ratio)

  # Création des matrices de train et test.
  train = np.array(examples[:n_train_examples])
  train_labels = np.array(labels[:n_train_examples])
  test = np.array(examples[n_train_examples:])
  test_labels = np.array(labels[n_train_examples:])
  
  # Tres important : la fonction doit retourner 4 matrices (ou vecteurs) de type Numpy. 
  return (train, train_labels, test, test_labels)

def normalize(dataset):
    """
    Applique une standardisation (z-score normalization) sur chaque colonne d'un array numpy à 2 dimensions.
    Ne modifie pas le dataset original. Renvoie une copie standardisée.
    
    Paramètres :
        - dataset : array numpy à 2 dimensions
        
    Retourne :
        - std_dataset : array numpy standardisé
    """
    # Calcul des moyennes et écart-types de chaque colonne
    moyennes = np.mean(dataset, axis=0)
    ecarts_types = np.std(dataset, axis=0)
    
    # Soustraction des moyennes pour centrer les données
    dataset_centre = dataset - moyennes
    
    # Division par les écart-types pour standardiser les données
    std_dataset = dataset_centre / ecarts_types
    
    return std_dataset

def load_iris_dataset(train_ratio):
    """Cette fonction a pour but de lire le dataset Iris

    Args:
        train_ratio: le ratio des exemples qui vont etre attribués à l'entrainement,
        le reste des exemples va etre utilisé pour les tests.
        Par exemple : si le ratio est 50%, il y aura 50% des exemple (75 exemples) qui vont etre utilisés
        pour l'entrainement, et 50% (75 exemples) pour le test.

    Retours:
        Cette fonction doit retourner 4 matrices de type Numpy: train, train_labels, test, et test_labels
		
        - train : une matrice numpy qui contient les exemples qui vont etre utilisés pour l'entrainement, chaque 
        ligne dans cette matrice représente un exemple d'entrainement.
		
        - train_labels : contient les labels (ou les étiquettes) pour chaque exemple dans train, de telle sorte
          que : train_labels[i] est l'etiquette pour l'exemple train[i]
        
        - test : une matrice numpy qui contient les exemples qui vont etre utilisés pour le test, chaque 
        ligne dans cette matrice représente un exemple de test.
		
        - test_labels : contient les étiquettes pour chaque exemple dans test, de telle sorte
          que : test_labels[i] est l'etiquette pour l'exemple test[i]
    """
    train, train_labels, test, test_labels = _load_dataset('datasets/bezdekIris.data', train_ratio, 5)
    
    conversion_labels = {'Iris-setosa': "setosa", 'Iris-versicolor' : "versicolor", 'Iris-virginica' : "virginica"}
    train_labels = np.array([conversion_labels[label] for label in train_labels])
    test_labels = np.array([conversion_labels[label] for label in test_labels])
    
    train, test = normalize(train), normalize(test)
    
    # Tres important : la fonction doit retourner 4 matrices (ou vecteurs) de type Numpy.
    return (train, train_labels, test, test_labels)


def load_wine_dataset(train_ratio):
    """Cette fonction a pour but de lire le dataset Binary Wine quality

    Args:
        train_ratio: le ratio des exemples (ou instances) qui vont servir pour l'entrainement,
        le reste des exemples va etre utilisé pour les test.

    Retours:
        Cette fonction doit retourner 4 matrices de type Numpy: train, train_labels, test, et test_labels
		
        - train : une matrice numpy qui contient les exemples qui vont etre utilisés pour l'entrainement, chaque 
        ligne dans cette matrice représente un exemple d'entrainement.
		
        - train_labels : contient les labels (ou les étiquettes) pour chaque exemple dans train, de telle sorte
          que : train_labels[i] est l'etiquette pour l'exemple train[i]
        
        - test : une matrice numpy qui contient les exemples qui vont etre utilisés pour le test, chaque 
        ligne dans cette matrice représente un exemple de test.
		
        - test_labels : contient les étiquettes pour chaque exemple dans test, de telle sorte
          que : test_labels[i] est l'etiquette pour l'exemple test[i]
    """
	
    train, train_labels, test, test_labels = _load_dataset('datasets/binary-winequality-white.csv', train_ratio, 12)
    train, test = normalize(train.astype(float)), normalize(test.astype(float))
    train_labels, test_labels = train_labels.astype(int), test_labels.astype(int)
    
    # La fonction doit retourner 4 structures de données de type Numpy.
    return (train, train_labels, test, test_labels)


def load_abalone_dataset(train_ratio):
    """
    Cette fonction a pour but de lire le dataset Abalone-intervalles

    Args:
        train_ratio: le ratio des exemples (ou instances) qui vont servir pour l'entrainement,
        le reste des exemples va etre utilisé pour les test.

    Retours:
        Cette fonction doit retourner 4 matrices de type Numpy: train, train_labels, test, et test_labels

        - train : une matrice numpy qui contient les exemples qui vont etre utilisés pour l'entrainement, chaque 
        ligne dans cette matrice représente un exemple d'entrainement.

        - train_labels : contient les labels (ou les étiquettes) pour chaque exemple dans train, de telle sorte
          que : train_labels[i] est l'etiquette pour l'exemple train[i]

        - test : une matrice numpy qui contient les exemples qui vont etre utilisés pour le test, chaque 
        ligne dans cette matrice représente un exemple de test.

        - test_labels : contient les étiquettes pour chaque exemple dans test, de telle sorte
          que : test_labels[i] est l'etiquette pour l'exemple test[i]
    """
    
    # conversion = {'M': (0,0,1), 'F' : (0,1,0), 'I' : (1,0,0)}
    conversion = {'M': 1.0, 'F' : 0.0, 'I' : 0.5}
    # La fonction doit retourner 4 matrices (ou vecteurs) de type Numpy.
    train, train_labels, test, test_labels = _load_dataset('datasets/abalone-intervalles.csv', train_ratio, 9)
    for row in train:
      row[0] = conversion[row[0]]
    for row in test:
      row[0] = conversion[row[0]]
    train, test = normalize(train.astype(float)), normalize(test.astype(float))
    train_labels, test_labels = train_labels.astype(float).astype(int), test_labels.astype(float).astype(int)
    return (train, train_labels, test, test_labels)


