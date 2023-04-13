# TP3: Implémentations des K-nn et de la classification naïve bayésienne

**Elliot Fontaine | Adam Bakry**



## Répartition des tâches:
- **Elliot**: classe `Knn(Classifier)`, `load_dataset.py`, `statistiques.py`, `README.md`, rapport PDF, architecture du projet.  
*TRANSPARENCE*: des outils basés sur l'intelligence artificielle (ChatGPT et Github Copilot) ont été utilisés pour le codage. Le filtre de suggestions de Copilot a été activé pour éviter de plagier du code public.
- **Adam**: classe `NaiveBayes(Classifier)`, rapport PDF

# Description de l'architecture du projet
- Les algorithmes de classification sont implémentés dans `Classifiers.py` (lire plus bas).
- `load_datasets.py` contient une fonction générique `_load_dataset` appelée par les fonctions spécifiques aux 3 jeux de données. Le data engineering se fait également dans ces fonctions (conversions de type, normalisation, etc...)
- Le fichier principal est `entrainer_tester.py`
- `statistiques.py` contient les fonctions permettant de calculer les métriques à partir d'une matrice de confusion.
- `comparaison-scikit-learn` est un fichier indépendant semblable au fichier principal, mais qui utilise le module scikit-learn pour le Knn.

## Description des classes dans *Classifiers.py*:
- Classe abstraite `Classifier`: expose trois méthodes `train`, `predict` et `evaluate`. Les deux premières sont abstraites et doivent être implémentées dans les classes filles. La dernière méthode `evaluate` est implémentée et calcule la matrice de confusion sur un jeu de données de test, en utilisant la méthode `predict` pour prédire le label de chaque vecteur de données.
- Classe `Knn(Classifier)`: implémente les méthodes `train` et `predict` de la classe abstraite `Classifier`. La méthode `train` place les données d'entrainement dans un tableau numpy, attribut d'instance. La méthode `predict` calcule la distance euclidienne entre un vecteur de données et chaque vecteur d'entrainement. Elle retourne le label majoritaire parmi les k plus proches voisins. 
- Classe `NaiveBayes(Classifier)`: implémente les méthodes `train` et `predict` de la classe abstraite `Classifier`. **[...]**

## Difficultés rencontrées:
- **Nature des données:** la conversion des données catégorielles non-binaires pour le K-nn. En particulier, le sexe (mâle, femelle, enfant) du datatset Abalones. On a décidé de convertir {M, I, F} en {O, 1, 2}, même si ca provoque des distances inégales entre les 3. Les résultats restent honorables pour ce dataset (exactitude=81%)
- **Implémentation du classificateur naïf bayésien:**
