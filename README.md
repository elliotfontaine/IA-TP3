# TP3: Implémentations de l'arbre de décision et du réseau de neurones

**Elliot Fontaine | Adam Bakry**



## Répartition des tâches:
- **Elliot**: classe `NeuralNet(Classifier)`, `load_dataset.py`, `statistiques.py`, `README.md`, `comparaison-scikit-learn`, rapport PDF, architecture du projet.  
*TRANSPARENCE*: des outils basés sur l'intelligence artificielle (ChatGPT et Github Copilot) ont été utilisés pour le codage. Le filtre de suggestions de Copilot a été activé pour éviter de plagier du code public.
- **Adam**: classe `DecisionTree(Classifier)`, rapport PDF

# Description de l'architecture du projet
- Les algorithmes de classification sont implémentés dans `Classifiers.py` (lire plus bas).
- `load_datasets.py` contient une fonction générique `_load_dataset` appelée par les fonctions spécifiques aux 3 jeux de données. Le data engineering se fait également dans ces fonctions (conversions de type, normalisation, etc...)
- Le fichier principal est `entrainer_tester.py`
- `statistiques.py` contient les fonctions permettant de calculer les métriques à partir d'une matrice de confusion.
- `comparaison-scikit-learn` est un fichier indépendant semblable au fichier principal, mais qui utilise le module scikit-learn pour comparer les résultats.

## Description des classes dans *Classifiers.py*:
- Classe abstraite `Classifier`: expose trois méthodes `train`, `predict` et `evaluate`. Les deux premières sont abstraites et doivent être implémentées dans les classes filles. La dernière méthode `evaluate` est implémentée et calcule la matrice de confusion sur un jeu de données de test, en utilisant la méthode `predict` pour prédire le label de chaque vecteur de données.
- Classe `NeuralNet(Classifier)`: implémente les méthodes `train` et `predict` de la classe abstraite `Classifier`. **[...]**
- Classe `DecisionTree(Classifier)`: implémente les méthodes `train`, `evaluate` et `predict` de la classe abstraite `Classifier`. **[...]**

## Difficultés rencontrées:
- **DecisionTree:** J'ai réussi a implémenter l'hyper paramètre de la hauteur de l'arbre, mais quand j'ai tenté de rajouté la gestion des variables continue et l'hyperparamètre qui gère le nombre de coupe et le nombre de valeur maximum pour une variable, le code a explosé et j'ai du tenté de sauvé ce que je pouvais pour la remise. A cause de manque de la gestion de variables continue, l'abre fait de l'overfitting extrème en plus d'avoir un bug ou il a presque toujours une hauteur de 1 ce qui donne des très mauvais résultat.
