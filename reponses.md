Un document PDF contenant :
    — Les réponses aux questions
    — Les discussions des résultats obtenus
    — Une comparaison entre les deux techniques d’apprentissage en terme de performances : Temps d’exécution, Accuracy, Precision et Recall. (Faites un tableau récapitulatif ).
    — Une conclusion (mentionnez aussi les difficultés rencontrées)




# REPONSES AUX QUESTIONS


1. Faites une recherche et définissez dans vos mots les termes suivants :
— La précision (Precision)
— Le rappel (Recall)
— Le F1-score
— La matrice de confusion (Confusion matrix). Dans votre définition, expliquez com- ment l’exactitude, la précision, le rappel et le F1-score peuvent tous être extraits à partir de la matrice de confusion.


1. Réponse par ChatGPT. À relire (vérifier l'exactitude)
En apprentissage automatique, plusieurs mesures sont utilisées pour évaluer les performances des modèles de classification, de régression ou de clustering. Voici les définitions des termes mentionnés :

- La précision (Precision) est une mesure de la qualité des prédictions positives d'un modèle. Elle est définie comme le rapport entre le nombre de vrais positifs (TP) et le nombre total de prédictions positives (TP + FP) : Precision = TP / (TP + FP). En d'autres termes, la précision mesure la proportion de prédictions positives qui sont effectivement positives.
- Le rappel (Recall) est une mesure de la sensibilité d'un modèle. Il mesure la proportion de vrais positifs qui ont été identifiés parmi tous les exemples positifs (vrais positifs + faux négatifs) : Recall = TP / (TP + FN). Le rappel mesure donc la capacité du modèle à identifier tous les exemples positifs.
- Le F1-score est une mesure de la performance globale d'un modèle, qui prend en compte à la fois la précision et le rappel. Il est défini comme la moyenne harmonique de la précision et du rappel : F1-score = 2 * (Precision * Recall) / (Precision + Recall). Le F1-score est particulièrement utile lorsque l'on veut trouver un compromis entre la précision et le rappel.
- La matrice de confusion (Confusion matrix) est une table qui permet de visualiser les prédictions d'un modèle en fonction des vraies étiquettes de chaque exemple. Elle est généralement présentée sous la forme d'un tableau à deux dimensions, où chaque ligne correspond à une vraie étiquette et chaque colonne correspond à une prédiction. Les quatre cellules de la matrice représentent les quatre résultats possibles : les vrais positifs (TP), les faux positifs (FP), les faux négatifs (FN) et les vrais négatifs (TN).

A partir de la matrice de confusion, il est possible de calculer plusieurs mesures de performance, dont l'exactitude, la précision, le rappel et le F1-score :
- L'exactitude (Accuracy) est définie comme le rapport entre le nombre de prédictions correctes (TP + TN) et le nombre total d'exemples (TP + FP + TN + FN) : Accuracy = (TP + TN) / (TP + FP + TN + FN). L'exactitude mesure la proportion d'exemples correctement classés par le modèle.
- La précision et le rappel peuvent être calculés à partir des cellules de la matrice de confusion : Precision = TP / (TP + FP) et Recall = TP / (TP + FN).
- Enfin, si on peut calculer la précision et le rappel, le F1-score peut donc également être calculé à partir des cellules de la matrice de confusion.

De plus, on pourrait rajouter la Spécificité = VN / (VN + FP). Cette dernière métrique n'est pas très utilisée en machine learning, mais elle est très importante en épidémiologie ou lors des tests cliniques, autre domaine d'utilisation des matrices de confusion.