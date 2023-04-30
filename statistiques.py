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
    Calcule le F1-score (f1_score) pour une des classes/labels à partir d'une matrice de confusion
    f1_score = 2 * (precision * recall) / (precision + recall)
    """
    p = class_precision(confusion_matrix, true_label)
    r = class_recall(confusion_matrix, true_label)
    return 2 * (p * r) / (p + r)

def macro_precision(confusion_matrix):
    """
    Calcule la précision moyenne à partir d'une matrice de confusion
    (moyenne arithmétique)
    """
    return np.mean([class_precision(confusion_matrix, label) for label in confusion_matrix])

def macro_recall(confusion_matrix):
    """
    Calcule le rappel moyen à partir d'une matrice de confusion
    (moyenne arithmétique)
    """
    return np.mean([class_recall(confusion_matrix, label) for label in confusion_matrix])

def macro_f1_score(confusion_matrix):
    """
    Calcule le F1-score moyen à partir d'une matrice de confusion.
    (moyenne arithmétique)
    
    /!\ Attention à la pertinence d'une telle statistique pour un problème à plusieurs classes
    https://towardsdatascience.com/multi-class-metrics-made-simple-part-ii-the-f1-score-ebe8b2c2ca1
    """
    return np.mean([class_f1_score(confusion_matrix, label) for label in confusion_matrix])

def weighted_precision(confusion_matrix):
    """
    Calcule la précision pondérée par les effectifs de classes à partir d'une matrice de confusion
    """
    weights = [sum([confusion_matrix[predicted_label][true_label] for predicted_label in confusion_matrix]) for true_label in confusion_matrix]
    class_precisions = [class_precision(confusion_matrix, label) for label in confusion_matrix]
    return np.average(class_precisions, weights=weights)

def weighted_recall(confusion_matrix):
    """
    Calcule le rappel pondéré par les effectifs de classes à partir d'une matrice de confusion
    """
    weights = [sum([confusion_matrix[predicted_label][true_label] for predicted_label in confusion_matrix]) for true_label in confusion_matrix]
    class_recalls = [class_recall(confusion_matrix, label) for label in confusion_matrix]
    return np.average(class_recalls, weights=weights)

def weighted_f1_score(confusion_matrix):
    """
    Calcule le F1-score pondéré par les effectifs de classes à partir d'une matrice de confusion
    """
    weights = [sum([confusion_matrix[predicted_label][true_label] for predicted_label in confusion_matrix]) for true_label in confusion_matrix]
    class_f1_scores = [class_f1_score(confusion_matrix, label) for label in confusion_matrix]
    return np.average(class_f1_scores, weights=weights)

def print_stats(confusion_matrix, binary=False):
    """
    Affiche les statistiques calculées à partir d'une matrice de confusion.
    
    Args:
        - confusion_matrix (dict): matrice de confusion issue d'un modèle de classification.
        - binary (bool): indique si le problème est binaire ou multi-classe.
        
    Dans le cas binaire, le rappel, la précision et le F1-score sont affichés pour la classe positive (d'étiquette 1).  
    Dans le cas multi-classe, le rappel, la précision et le F1-score sont calculés pour chaque classe et les moyennes (macro)
    et moyennes pondérées par les effectifs de classes (weighted) sont affichées.
    """
    print("Matrice de confusion: ", confusion_matrix)
    print("exactitude:  ", round(accuracy(confusion_matrix), 2))
    if binary:
        for stat in [class_precision, class_recall, class_f1_score]:
            try:
                print(stat.__name__, ": ", round(stat(confusion_matrix, 1), 2))
            except ZeroDivisionError:
                print(stat.__name__, ": ", "Non défini (division par 0)")
    else:
        for stat in [macro_precision, macro_recall, macro_f1_score, weighted_precision, weighted_recall, weighted_f1_score]:
            try:
                print(stat.__name__, ": ", round(stat(confusion_matrix), 2))
            except ZeroDivisionError:
                print(stat.__name__, ": ", "Non défini (division par 0)")
            
        
    