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
from abc import ABC, abstractmethod

# le nom de votre classe
# BayesNaif pour le modèle bayesien naif
# Knn pour le modèle des k plus proches voisins

class Classifier(ABC): #nom de la class à changer
	
	@abstractmethod
	def train(self, train, train_labels): #vous pouvez rajouter d'autres attributs au besoin
		"""
		C'est la méthode qui va entrainer votre modèle,
		train est une matrice de type Numpy et de taille nxm, avec 
		n : le nombre d'exemple d'entrainement dans le dataset
		m : le mobre d'attribus (le nombre de caractéristiques)
		
		train_labels : est une matrice numpy de taille nx1
		
		vous pouvez rajouter d'autres arguments, il suffit juste de
		les expliquer en commentaire
		
		"""
		pass
    
	@abstractmethod
	def predict(self, x):
		"""
		Prédire la classe d'un exemple x donné en entrée
		exemple est de taille 1xm
		"""
		pass
        
	def evaluate(self, X, y, labels, **kwargs):
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
			prediction = self.predict(X[i], **kwargs)
			confusion_matrix[prediction][y[i]] += 1
		return confusion_matrix
 
class Knn(Classifier):

	def __init__(self, k=5, train_data=None, train_labels=None):
		"""
		C'est un Initializer. 
		Vous pouvez passer d'autre paramètres au besoin,
		c'est à vous d'utiliser vos propres notations
		"""
		self.k = k
		self.train_data = train_data
		self.train_labels = train_labels
        
        
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
    
	def predict(self, x: np.ndarray, distance_type='euclidean'):
		"""
		Prédire la classe d'un exemple x donné en entrée
		exemple est de taille 1xm
		"""
		# Préconditions
		if self.train_data is None or self.train_labels is None:
			raise ValueError('Train data or labels not set')
		if self.k > len(self.train_data):
			raise ValueError('k is greater than the number of training examples')
		# Calcul des distances
		if distance_type == 'euclidean':
			distances = np.sqrt(np.sum((self.train_data - x)**2, axis=1))
		elif distance_type == 'manhattan':
			distances = np.sum(np.abs(self.train_data - x), axis=1)
		else:
			raise ValueError('Distance type not recognized')
		# Prédiction du label
		k_nearest = np.argsort(distances)[:self.k]
		k_nearest_labels = self.train_labels[k_nearest].tolist()
		# return (np.bincount(k_nearest_labels)).argmax()
		most_frequent_label = max(set(k_nearest_labels), key = k_nearest_labels.count)
		return most_frequent_label
        
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
		return super().evaluate(X, y, labels, distance_type=distance_type)
		
class NaiveBayes(Classifier):

	def __init__(self, train_data=None, train_labels=None):
		"""
		C'est un Initializer. 
		Vous pouvez passer d'autre paramètres au besoin,
		c'est à vous d'utiliser vos propres notations
		"""
		pass