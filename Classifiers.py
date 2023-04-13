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
import math
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
		
		Retourne:
			confusion_matrix: 
   			Exemple de matrice de confusion :
  			{A: {A: 11, B: 0, C: 0}, B: {A: 0, B: 8, C: 0}, C: {A: 0, B: 1, C: 10}} où A, B et C sont les labels des classes
			On la parcourt de la manière suivante: confusion_matrix[prediction][vrai_label]
		"""
		confusion_matrix = {l: {l: 0 for l in labels} for l in labels}
		for i in range(len(X)):
			prediction = self.predict(X[i], **kwargs)
			confusion_matrix[prediction][y[i]] += 1
		return confusion_matrix
 
class Knn(Classifier):

	def __init__(self, k, train_data=None, train_labels=None):
		"""
		C'est un Initializer. 
		Vous pouvez passer d'autre paramètres au besoin,
		c'est à vous d'utiliser vos propres notations
		"""
		self.k = k
		self.train_data = train_data
		self.train_labels = train_labels
		self.unique_labels = np.unique(train_labels)
        
        
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
		self.unique_labels = np.unique(train_labels)
    
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
        
	def evaluate(self, X, y, distance_type='euclidean'):
		"""
		c'est la méthode qui va evaluer votre modèle sur les données X
		l'argument X est une matrice de type Numpy et de taille nxm, avec 
		n : le nombre d'exemple de test dans le dataset
		m : le mobre d'attribus (le nombre de caractéristiques)
		
		y : est une matrice numpy de taille nx1
		
		vous pouvez rajouter d'autres arguments, il suffit juste de
		les expliquer en commentaire
		"""
		matrix_labels = np.unique(np.concatenate((self.train_labels, y)))
		return super().evaluate(X, y, matrix_labels, distance_type=distance_type)
		
class NaiveBayes(Classifier):

	def __init__(self, train_data=None, train_labels=None, validation_data=None, validation_labels=None, tree=None):
		"""
		C'est un Initializer. 
		Vous pouvez passer d'autre paramètres au besoin,
		c'est à vous d'utiliser vos propres notations
		"""
		self.train_data = train_data
		self.train_labels = train_labels
		self.validation_data = validation_data
		self.validation_labels = validation_labels
		self.tree = tree

	def train(self, train, train_labels, att_index, default): #vous pouvez rajouter d'autres attributs au besoin
		"""
		C'est la méthode qui va entrainer votre modèle,
		train est une matrice de type Numpy et de taille nxm, avec 
		n : le nombre d'exemple d'entrainement dans le dataset
		m : le mobre d'attribus (le nombre de caractéristiques)
		
		train_labels : est une matrice numpy de taille nx1
		
		vous pouvez rajouter d'autres arguments, il suffit juste de
		les expliquer en commentaire
		
		"""
		
		# séparer train en train et validation
		self.generate_validation(train, train_labels)

		# générer l'Arbre
		av_length = len(att_index)
		self.tree = self.build_tree(train, train_labels, att_index, default, [1] * av_length)

		#valider l'arbre


		
		

	def predict(self, x):
		"""
		Prédire la classe d'un exemple x donné en entrée
		exemple est de taille 1xm
		"""
		tree = self.tree
		
        

	def evaluate(self, X, y, labels):
		"""
		c'est la méthode qui va evaluer votre modèle sur les données X
		l'argument X est une matrice de type Numpy et de taille nxm, avec 
		n : le nombre d'exemple de test dans le dataset
		m : le mobre d'attribus (le nombre de caractéristiques)
		
		y : est une matrice numpy de taille nx1
		
		vous pouvez rajouter d'autres arguments, il suffit juste de
		les expliquer en commentaire
		"""
		return super().evaluate(X, y, labels)


	def generate_validation(self, train, train_labels):
		"""
		méthode qui permet de crée validation et validation_labels
		en retirant un sous ensemble de train et train_labels  
		"""
		self.train_data = []
		self.train_labels = []
		self.validation_data = []
		self.validation_labels = []

		one_tenth = 0.1 * len(train)

		#mettre les premier 90% dans train
		for i in range (len(train) - one_tenth):
			self.train_data.append(self.train_data[i])
			self.train_labels.append(self.train_labels[i])
		
		#mettre les 10% dernier dans validation
		for i in range(len(train) - one_tenth, len(train)):
			self.validation_data.append(self.train_data[i])
			self.validation_labels.append(self.train_labels[i])
			
	
	def get_frequencies(self, data_array):
		"""
		prend un array de données et retourne le dictionnaire 
		des fréquences des différentes valeurs présentes
		dans les données de l'array.
		"""
		frequencies = {}

		# trouver les fréquence
		for ex in data_array:
			if ex in frequencies:
				frequencies[ex] = frequencies[ex] + 1
			else:
				frequencies[ex] = 1

		return frequencies


	def build_tree(self, train, train_labels, att_index, default, av_index):
		"""
		construire l'arbre de décision a l'aide d'appel récursifes

		train: l'ensemble d'exemples dans un array d'array
		train_labels: array des labels pour les exemples
		att_index: un array de l'index des attributs disponible
		av_index
		default: la valeur par défaut a retourner
		"""
		av_att = 0
		for av in av_index:
			if av == 1:
				av_att += 1
		if len(train) == 0 : 
			return default
		elif len(self.get_frequencies(train_labels)) == 1 :
			return train_labels[0]
		elif av_att == 0 :
			return self.majority_value(train_labels)
		else:
			diminished_index = []
			for i in range(len(av_index)):
				if av_index[i] == 1:
					diminished_index.append(att_index[i])

			best_att = self.choose_attribut(diminished_index, train_labels, train)
			best_att_diminished_index = diminished_index.index(best_att)
			temps_position = -1
			for i in range(len(av_index)) :
				if av_index[i] == 1:
					temps_position += 1
				if temps_position == best_att_diminished_index:
					best_att_index = i
					break
			
			best_att_array = train[best_att_index]
			best_att_freq = self.get_frequencies(best_att_array)
			tree = [best_att_index]
			majority = self.majority_value(train_labels)
			new_av_index = av_index
			for i in range(len(att_index)):
				if att_index[i] == att_index[best_att_index]:
					new_av_index[i] = 0
			branches = []
					
			for p_value in best_att_array:
				# obtenir les arrays des exemples updater avec la valeur = de lattribut = p_value
				temp_train = []
				for i in range(av_att):
					temp_train.append([])
				temp_labels = [] 
				for i in range(len(train[best_att_index])):
					if best_att_array[i] == p_value :
						for j in range(av_att):
							temp_train[j].append(train[j][i])
						temp_labels.append(train_labels[i])
			
				#crée le sous arbre
				sub_tree = self.build_tree(temp_train, temp_labels, att_index, majority, new_av_index)

				#ajouter les branches avec sub_tree
				branche = [p_value, sub_tree]
				branches.append(branche)
			
			tree.append(branches)
			
			return tree

	
	def majority_value(self, labels_data):
		"""
		retourne l'étiquette ayant la plus grande fréquence dans l'array labels_data.

		labels_data: array des labels 
		"""
		frequencies = self.get_frequencies(labels_data)
		max_val = max(frequencies, key= frequencies.get)

		return max_val

	
	def choose_attribut(self, attributes, labels_data, attributes_data_matrice):
		"""
		retourne l'index de l'attribut ayant le plus grand gain ratio dans la liste d'exemples.

		attributs: array de l'index des attributs que l'on peut choisir
		labels_data: array des labels des exemples
		attribute_data_matrice: array d'array des valeurs de chaque attribut des exemples
		"""

		attributes_gain_ratio = {} # dictionnaire ayant le gain ratio pour chaque attribut
		
		for attribute in attributes: #remplis le dictionnaire
			attribute_data_array = attributes_data_matrice[attribute]
			gain_ratio = self.get_gain_ratio(labels_data, attribute_data_array)
			attributes_gain_ratio[attribute] = gain_ratio

		return max(attributes_gain_ratio, key= attributes_gain_ratio.get)
		

	def get_gain_ratio(self, labels_data, attribute_data):
		"""
		retourne le gain ratio pour l'attribut représenté par attribute_data.

		labels_data: array des labels des exemples
		attribute_data: array des valeurs de l'attribut des exemples
		"""

		attribute_frequencies = self.get_frequencies(attribute_data)

		gain = self.get_gain(labels_data, attribute_data)
		
		#calculer split information sum
		temp_sum = 0
		for key in attribute_frequencies:
			fraction = attribute_frequencies[key] / len(labels_data)
			temp_sum += fraction * math.log2(fraction)

		split_information = 0 - temp_sum

		return gain / split_information


	def get_gain(self, labels_data, attribute_data):
		"""
		retourne le gain relier l'attribut représenté par attribute_data.

		labels_data: array des labels des exemples
		attribute_data: array des valeurs de l'attribut des exemples
		"""
		attribute_frequencies = self.get_frequencies(attribute_data)
		attribute_amount = len(attribute_frequencies)
		
		#l'entropie des labels (Entropie(S))
		labels_entropie = self.get_label_entropy(labels_data)

		# obtentenir dictionaire l'entropie de chaque valeur de l'attribut
		attribut_entropie = self.get_entropy(attribute_data, labels_data)
		
		# calculer la somme de l'expression du gain
		entropie_sum = 0
		for key in attribut_entropie:
			sv_amount = attribute_frequencies[key]
			entropie_sum += sv_amount / len(labels_data) * attribut_entropie[key]

		return labels_entropie - entropie_sum


	def get_entropy(self, attribute_data, labels_data):
		"""
		retourne un dictionaire de l'entropie pour chaque valeur d'un seul attribut.

		attribute_data = array des valeurs d'un attributs des exemples.
		labels_data = array des labels.
		"""
		frequencies = self.get_frequencies(attribute_data) #contient les frequences total
		label_frequencies = self.get_frequencies(labels_data)
		combined = []
		label_freq = {} #key = valeurs possible de l'attribut, value = array des fréquences selon les labels de cette valeur dans les exemples
		entropies = {}
		values = [] #valeurs possible des etiquettes

		for key in label_frequencies:	#iter sur frequencies pour remplir values
			values.append(key)
		
		for i in range(len(attribute_data)): #iter a travers attribute_data et labels_data en meme temps pour remplir label_freq
			combined.append([attribute_data[i], labels_data[i]])

			if combined[i][0] in label_freq: 
				temp = label_freq[combined[i][0]]		
				temp[values.index(combined[i][1])] = temp[values.index(combined[i][1])] + 1
				label_freq[combined[i][0]] = temp
			else:
				temp = [0] * len(values)	
				temp[values.index(combined[i][1])] = temp[values.index(combined[i][1])] + 1
				label_freq[combined[i][0]] = temp

		for p_value in label_freq:
			entropie = 0
			for numerator in label_freq[p_value]:
				if numerator != 0:
					fraction = numerator / frequencies[p_value]
					negative = 0 - fraction
					entropie += negative * math.log2(fraction)
			
			entropies[p_value] = entropie
	
		return entropies


	def get_label_entropy(self, labels_data):
		"""
		retourne l'entropie pour les étiquettes seulement.

		labels_data = array des labels.
		"""
		frequencies = self.get_frequencies(labels_data)
		entropie = 0
		total = 0
		for key in frequencies:
			total += frequencies[key]
		for key in frequencies:
			fraction = frequencies[key] / total
			negative = 0 - fraction
			entropie += negative * math.log2(fraction)

		return entropie

"""" #get_label_entropy
bells = ["baba", "bobo", "baba", "baba", "bobo", "baba", "baba", "bobo", "baba", "baba", "bobo", "baba", "baba", "bobo"]
b = NaiveBayes()
x = b.get_label_entropy(bells)
print(x)
"""
""" #get_entropy
att2 = ['e', 'e', 'n', 'p', 'p', 'p', 'n', 'e', 'e', 'p', 'e', 'n', 'n', 'p']
bells2 = [0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0]
b = NaiveBayes()
x2 = b.get_entropy(att2, bells2)
print(x2)
"""

"""#get_gain
att = ['e', 'e', 'n', 'p', 'p', 'p', 'n', 'e', 'e', 'p', 'e', 'n', 'n', 'p']
bells = [0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0]
b = NaiveBayes()
x = b.get_gain(bells, att)
print(x)
"""

"""#get_gain_ratio
att = ['e', 'e', 'n', 'p', 'p', 'p', 'n', 'e', 'e', 'p', 'e', 'n', 'n', 'p']
bells = [0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0]
b = NaiveBayes()
x = b.get_gain_ratio(bells, att)
print(x)
"""

"""#choose_attribute
att1 = ['e', 'e', 'n', 'p', 'p', 'p', 'n', 'e', 'e', 'p', 'e', 'n', 'n', 'p'] #ciel
att2 = ['c', 'c', 'c', 't', 'f', 'f', 'f', 't', 'f', 't', 't', 't', 'c', 't'] #temp
att3 = ['e', 'e', 'e', 'e', 'n', 'n', 'n', 'e', 'n', 'n', 'n', 'e', 'n', 'e'] #humidite
att4 = ['w', 's', 'w', 'w', 'w', 's', 's', 'w', 'w', 'w', 's', 's', 'w', 's'] #vent
matrice = [att1, att2, att3, att4]
bells = [0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0] #jouerTennis
indexes = [0,1,2,3]
b = NaiveBayes()
x = b.choose_attribut(indexes, bells, matrice)
print(x)
"""


""" #build_tree
att1 = ['e', 'e', 'n', 'p', 'p', 'p', 'n', 'e', 'e', 'p', 'e', 'n', 'n', 'p'] #ciel
att2 = ['c', 'c', 'c', 't', 'f', 'f', 'f', 't', 'f', 't', 't', 't', 'c', 't'] #temp
att3 = ['e', 'e', 'e', 'e', 'n', 'n', 'n', 'e', 'n', 'n', 'n', 'e', 'n', 'e'] #humidite
att4 = ['w', 's', 'w', 'w', 'w', 's', 's', 'w', 'w', 'w', 's', 's', 'w', 's'] #vent
matrice = [att1, att2, att3, att4]
bells = [0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0] #jouerTennis
indexes = [0,1,2,3]
b = NaiveBayes()
x = b.build_tree(matrice, bells, indexes, bells[0], [1,1,1,1]) """
