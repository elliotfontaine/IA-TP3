"""
Vous allez definir une classe pour chaque algorithme que vous allez développer,
votre classe doit contenir au moins les 3 méthodes definies ici bas, 
	* train 	: pour entrainer le modèle sur l'ensemble d'entrainement.
	* predict 	: pour prédire la classe d'un exemple donné.
	* evaluate 		: pour evaluer le classifieur avec les métriques demandées. 
vous pouvez rajouter d'autres méthodes qui peuvent vous etre utiles, mais la correction
se fera en utilisant les méthodes train, predict et evaluate de votre code.
"""

import numpy as np


# le nom de votre classe
# Knn pour le modèle des k plus proches voisins

class Knn: #nom de la class à changer

	def __init__(self, k=5):
		"""
		C'est un Initializer. 
		Vous pouvez passer d'autre paramètres au besoin,
		c'est à vous d'utiliser vos propres notations
		"""
		self.k = k
		self.train_data = None
		self.train_labels = None
        
        
	def train(self, train, train_labels): #vous pouvez rajouter d'autres attributs au besoin
		"""
		C'est la méthode qui va entrainer votre modèle,
		train est une matrice de type Numpy et de taille nxm, avec 
		n : le nombre d'exemple d'entrainement dans le dataset
		m : le nombre d'attributs (le nombre de caractéristiques)
		
		train_labels : est une matrice numpy de taille nx1
		
		vous pouvez rajouter d'autres arguments, il suffit juste de
		les expliquer en commentaire
		
		"""
		self.train_data = train
		self.train_labels = train_labels
    
	def predict(self, x, distance_type='euclidean'):
		"""
		Prédire la classe d'un exemple x donné en entrée
		exemple est de taille 1xm
		"""
		if distance_type == 'euclidean':
			distances = np.sqrt(np.sum((self.train_data - x)**2, axis=1))
		elif distance_type == 'manhattan':
			distances = np.sum(np.abs(self.train_data - x), axis=1)
		else:
			raise ValueError('Distance type not recognized')
		k_nearest = np.argsort(distances)[:self.k]
		k_nearest_labels = self.train_labels[k_nearest]
		return (np.bincount(k_nearest_labels)).argmax()
        
	def evaluate(self, X, y, labels, distance_type='euclidean'):
		"""
		c'est la méthode qui va evaluer votre modèle sur les données X
		l'argument X est une matrice de type Numpy et de taille nxm, avec 
		n : le nombre d'exemple de test dans le dataset
		m : le mobre d'attribus (le nombre de caractéristiques)
		
		y : est une matrice numpy de taille nx1
		
		vous pouvez rajouter d'autres arguments, il suffit juste de
		les expliquer en commentaire
		"""
		confusion_matrix = {l: {l: 0 for l in labels} for l in labels}
		for i in range(len(X)):
			prediction = self.predict(X[i], distance_type)
			confusion_matrix[prediction][y[i]] += 1
		return confusion_matrix
		

        
	
	# Vous pouvez rajouter d'autres méthodes et fonctions,
	# il suffit juste de les commenter.