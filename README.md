*README : un fichier texte contenant une brève description des classes, la répartition
des tâches de travail entre les membres d’équipe, et une explication des difficultés
rencontrées dans ce travail*

TODO:
- Utiliser la supposition que les labels sont dans la dernière colonne pour ne plus utiliser le paramètre lenght_data de load_dataset()
- Comparaison scikit-learn
- Normalisation données

**Répartition des tâches:**
- Elliot: classe `Knn(Classifier)`, `load_dataset.py`, `statistiques.py`, `README.md`, rapport PDF, architecture du projet.
- Adam: classe `NaiveBayes(Classifier)`, **[...]**

**Description des classes:**
- Classe abstraite `Classifier`: expose trois méthodes `train`, `predict` et `evaluate`. Les deux premières sont abstraites et doivent être implémentées dans les classes filles. La dernière méthode `evaluate` est implémentée et calcule la matrice de confusion sur un jeu de données de test, en utilisant la méthode `predict` pour prédire le label de chaque vecteur de données.
- Classe `Knn(Classifier)`: implémente les méthodes `train` et `predict` de la classe abstraite `Classifier`. La méthode `train` place les données d'entrainement dans un tableau numpy, attribut d'instance. La méthode `predict` calcule la distance euclidienne entre un vecteur de données et chaque vecteur d'entrainement. Elle retourne le label majoritaire parmi les k plus proches voisins. 
- Classe `NaiveBayes(Classifier)`: implémente les méthodes `train` et `predict` de la classe abstraite `Classifier`. **[...]**

**Difficultés rencontrées:**
- *Nature des données:* la conversion des données catégorielles non-binaires pour le K-nn. En particulier, le sexe (mâle, femelle, enfant) du datatset Abalones.
- **[...]**
