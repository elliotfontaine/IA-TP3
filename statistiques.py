"""
A partir d'une matrice de confusion formattée comme dictionnaire de dictionnaires, calcule les métriques suivantes :
- Exactitude (accuracy),
- Précision (precision),
- Rappel (recall),
- F1-score (f1_score)

Exemple de matrice de confusion :
  {A: {A: 11, B: 0, C: 0}, B: {A: 0, B: 8, C: 0}, C: {A: 0, B: 1, C: 10}} où A, B et C sont les labels des classes
  On la parcourt de la manière suivante: confusion_matrix[prediction][vrai_label]
"""

import numpy as np

def accuracy(confusion_matrix):
    """
    Calcule l'exactitude (accuracy) à partir d'une matrice de confusion
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    """
    T, all = 0, 0
    for label in confusion_matrix:
        T += confusion_matrix[label][label]
        all += sum(confusion_matrix[label].values())
    return T / all

def class_precision(confusion_matrix, true_label):
    """
    Calcule la précision (precision) pour une des classes/labels à partir d'une matrice de confusion
    precision = TP / (TP + FP)
    """
    TP = confusion_matrix[true_label][true_label]
    predicted_P = sum(confusion_matrix[true_label].values())
    return TP / predicted_P

def class_recall(confusion_matrix, true_label):
    """
    Calcule le rappel (recall) pour une des classes/labels à partir d'une matrice de confusion
    recall = TP / (TP + FN)
    """
    TP = confusion_matrix[true_label][true_label]
    actual_P = sum([confusion_matrix[label][true_label] for label in confusion_matrix])
    return TP / actual_P

def class_f1_score(confusion_matrix, true_label):
    """
    Calcule le F1-score (f1_score) à partir d'une matrice de confusion
    f1_score = 2 * (precision * recall) / (precision + recall)
    """
    p = class_precision(confusion_matrix, true_label)
    r = class_recall(confusion_matrix, true_label)
    return 2 * (p * r) / (p + r)

def mean_precision(confusion_matrix):
    """
    Calcule la précision moyenne à partir d'une matrice de confusion
    """
    return np.mean([class_precision(confusion_matrix, label) for label in confusion_matrix])

def mean_recall(confusion_matrix):
    """
    Calcule le rappel moyen à partir d'une matrice de confusion
    """
    return np.mean([class_recall(confusion_matrix, label) for label in confusion_matrix])

def mean_f1_score(confusion_matrix):
    """
    Calcule le F1-score moyen à partir d'une matrice de confusion.
    
    /!\ Attention à la pertinence d'une telle statistique pour un problème à plusieurs classes
    https://towardsdatascience.com/multi-class-metrics-made-simple-part-ii-the-f1-score-ebe8b2c2ca1
    """
    return np.mean([class_f1_score(confusion_matrix, label) for label in confusion_matrix])
