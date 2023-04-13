# TP3: Implémentations des K-nn et de la classification naïve bayésienne

**Elliot Fontaine | Adam Bakry**



## Répartition des tâches:
- **Elliot**: classe `Knn(Classifier)`, `load_dataset.py`, `statistiques.py`, `README.md`, rapport PDF, architecture du projet.  
*TRANSPARENCE*: des outils basés sur l'intelligence artificielle (ChatGPT et Github Copilot) ont été utilisés pour le codage. Le filtre de suggestions de Copilot a été activé pour éviter de plagier du code public.
- **Adam**: classe `NaiveBayes(Classifier)`, rapport PDF

## Description des classes dans *Classifiers.py*:
- Classe abstraite `Classifier`: expose trois méthodes `train`, `predict` et `evaluate`. Les deux premières sont abstraites et doivent être implémentées dans les classes filles. La dernière méthode `evaluate` est implémentée et calcule la matrice de confusion sur un jeu de données de test, en utilisant la méthode `predict` pour prédire le label de chaque vecteur de données.
- Classe `Knn(Classifier)`: implémente les méthodes `train` et `predict` de la classe abstraite `Classifier`. La méthode `train` place les données d'entrainement dans un tableau numpy, attribut d'instance. La méthode `predict` calcule la distance euclidienne entre un vecteur de données et chaque vecteur d'entrainement. Elle retourne le label majoritaire parmi les k plus proches voisins. 
- Classe `NaiveBayes(Classifier)`: implémente les méthodes `train` et `predict` de la classe abstraite `Classifier`. **[...]**

## Difficultés rencontrées:
- **Nature des données:** la conversion des données catégorielles non-binaires pour le K-nn. En particulier, le sexe (mâle, femelle, enfant) du datatset Abalones.
- **Implémentation du classificateur naïf bayésien:**
