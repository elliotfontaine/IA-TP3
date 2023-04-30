"""
Vous allez definir une classe pour chaque algorithme que vous allez développer,
votre classe doit contenir au moins les 3 méthodes definies ici bas, 
	* train 	: pour entraîner le modèle sur l'ensemble d'entrainement.
	* predict 	: pour prédire la classe d'un exemple donné.
	* evaluate 		: pour evaluer le classifieur avec les métriques demandées. 
vous pouvez rajouter d'autres méthodes qui peuvent vous etre utiles, mais la correction
se fera en utilisant les méthodes train, predict et evaluate de votre code.
"""

from abc import ABC, abstractmethod
import numpy as np
import math
import random
import statistiques as stat # importer les fonctions de calcul des métriques REMOVE


# le nom de votre classe
# DecisionTree pour l'arbre de décision
# NeuralNet pour le réseau de neurones

class Classifier(ABC): #nom de la class à changer
	
	@abstractmethod
	def train(self, train: np.ndarray, train_labels: np.ndarray): #vous pouvez rajouter d'autres attributs au besoin
		"""
		C'est la méthode qui va entrainer votre modèle,
		train est une matrice de type Numpy et de taille nxm, avec 
		n : le nombre d'exemple d'entrainement dans le dataset
		m : le nombre d'attributs (le nombre de caractéristiques)
		
		train_labels : est une matrice numpy de taille nx1
		
		vous pouvez rajouter d'autres arguments, il suffit juste de
		les expliquer en commentaire
		
		"""
		pass
    
	@abstractmethod
	def predict(self, x: np.ndarray):
		"""
		Prédire la classe d'un exemple x donné en entrée
		exemple est de taille 1xm
		"""
		pass
        
	def evaluate(self, X: np.ndarray, y: np.ndarray, labels: np.ndarray, **kwargs):
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
			result = self.predict(X[i], **kwargs)
			prediction = result if type(result) is not tuple else result[0]
			confusion_matrix[prediction][y[i]] += 1
		return confusion_matrix   
	
	# Vous pouvez rajouter d'autres méthodes et fonctions,
	# il suffit juste de les commenter.
 
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
    
	def predict(self, x, distance_type='euclidean'):
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


class DecisionTree(Classifier):

	def __init__(self, train_data=None, train_labels=None, validation_data=None, validation_labels=None, tree=None, hauteur_max=None, nb_coupe=None, nb_val_max=None, intervales=[]):
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
		self.hauteur_max = hauteur_max
		self.nb_coupe= nb_coupe
		self.nb_val_max = nb_val_max
		self.intervales = intervales

	def train(self, np_train, np_train_labels, att_index, max_maximum_height, max_nb_coupe, max_nb_val): #vous pouvez rajouter d'autres attributs au besoin
		"""
		C'est la méthode qui va entrainer votre modèle,
		train est une matrice de type Numpy et de taille nxm, avec 
		n : le nombre d'exemple d'entrainement dans le dataset
		m : le mobre d'attribus (le nombre de caractéristiques)
		
		train_labels : est une matrice numpy de taille nx1
		max_maximum_height : maximum de l'hyper paramètre hauteur
		max_nb_val : nombre max de valeur possible pour un attribut. affect le nombre 
					de point d'inflection quand on coupe des valeurs continue en discrètes
		
		vous pouvez rajouter d'autres arguments, il suffit juste de
		les expliquer en commentaire
		
		"""
		


		#conversion de numpymatrix a list
		train = np_train.transpose()
		train = train.tolist()
		train_labels = np_train_labels.transpose()
		train_labels = train_labels.tolist()
		
		#initialiser variable
		self.nb_val_max = max_nb_val


		

		#calcule de default pour generer l'arbre
		default = max(set(train_labels), key= train_labels.count)

		

		# générer l'Arbre
		av_length = len(att_index)

		#hyper param
		evaluations = []
		for i in range(1, max_maximum_height + 1 ):
			evaluations.append([])
			for j in range(1, max_nb_coupe + 1):
				self.nb_coupe = j
				self.intervales = []
				for k in range(len(train)):
					self.intervales.append([])

				# séparer train en train et validation
				self.generate_validation(train, train_labels, 0.2)
				temp_train = self.reduire_nb_valeur(self.train_data, self.train_labels)

				self.tree = self.build_tree(temp_train, self.train_labels, att_index, default, [1] * av_length, i)
				eval = self.evaluate(self.validation_data, self.validation_labels)
				evaluations[i - 1].append(stat.accuracy(eval))
			
		val_optimal = -1
		for i in range(len(evaluations)):
			for j in range(len(evaluations[i])):
				if evaluations[i][j] > val_optimal:
					val_optimal = evaluations[i][j]
					self.hauteur_max = i + 1
					self.nb_coupe = j + 1
		
		#generer arbre optimal
		temp_train = self.reduire_nb_valeur(train, train_labels)
		self.generate_validation(temp_train, train_labels, 0.2)
		self.tree = self.build_tree(self.train_data, self.train_labels, att_index, default, [1] * av_length, self.hauteur_max)

		#valider l'arbre


		
		

	def predict(self, x):
		"""
		Prédire la classe d'un exemple x donné en entrée
		exemple est de taille 1xm
		"""

		"""exemple de structure d'arbre pour l'exemple test. 0,2,3 sont les index, e,n,p,w,s sont des valeurs des attributs au index
		[0, 
    		[e, 
				[2, 
					[e,0],
					[n,1]
					]
				],
			[n, 1],
			
			[p, 
				[3,
					[w,1],
					[s,0]
				]
			]
		]
		"""
		answer = None
		tree = self.tree
		#tant que j'ai pas trouveer la réponse dans une feuille:
		while answer == None:
			if type(tree[1]) == type(tree):
				#traitement de noeud
				node_att_index = tree[0] #nous donne l'index de l'attribut a reguarder dans l'arbre
				x_node_value = x[node_att_index]
				correct_branche = None

				#pour chaque valeur possible de l'attribut node_att_index, je cherche la bonne branche
				for i in range(len(tree[1])): 
					if x_node_value == tree[1][i][0]:
						correct_branche = i
						break
				
				#changer arbre pour le sous arbre a visité
				temp_tree = tree[1][correct_branche][1]
				if type(temp_tree) == type([0,1]):
					tree = tree[1][correct_branche][1]
				
				else: #cas ou on est rendu a une feuille
					tree = tree[1][correct_branche]

			else:
				#traitement de feuille
				answer = tree[1]
				
		return answer
        

	def evaluate(self, X, y):
		"""
		c'est la méthode qui va evaluer votre modèle sur les données X
		l'argument X est une matrice de type Numpy et de taille nxm, avec 
		n : le nombre d'exemple de test dans le dataset
		m : le mobre d'attribus (le nombre de caractéristiques)
		
		y : est une matrice numpy de taille nx1
		
		vous pouvez rajouter d'autres arguments, il suffit juste de
		les expliquer en commentaire
		"""
		matrice = np.array(X)
		matrice = matrice.transpose()
		bels = np.array(y)
		
		
		#vérifier si il faut réduire et réduire les attributs
		reduced_train = np.array([])
		for i in range(len(matrice)):
			attribute_arr = matrice[i]
			if self.intervales[i] != float("-inf"):
					attribute_arr = self.reduire_array(matrice[i], bels, self.intervales[i])
					
			reduced_train.append(attribute_arr)
		
		reduced_train = reduced_train.transpose()
		
		labels = np.unique(np.concatenate((self.train_labels, y)))

		confusion_matrix = {l: {l: 0 for l in labels} for l in labels}
		for i in range(len(matrice)):
			exemple = []
			for j in range(len(X)):
				exemple.append(X[j][i])
			prediction = self.predict(exemple)
			confusion_matrix[prediction][bels[i]] += 1
		return confusion_matrix


	def generate_validation(self, train, train_labels, percentage):
		"""
		méthode qui permet de crée validation et validation_labels
		en retirant un sous ensemble de train et train_labels  
		"""
		self.train_data = []
		for i in range(len(train)):
			self.train_data.append([])
		self.train_labels = []
		self.validation_data = []
		for i in range(len(train)):
			self.validation_data.append([])
		self.validation_labels = []

		#en fait one_tenth n'est pas le nombre de cas représentant 1/10, mais bien selon la variable percentage
		one_tenth = round(percentage * len(train_labels))

		#mélanger l'ordre
		rand_order = list(range(0, len(train_labels)))
		random.shuffle(rand_order)
		rand_order1 = rand_order[0 : (len(train_labels) - one_tenth)]
		rand_order2 = rand_order[(len(train_labels) - one_tenth) : len(train_labels)]

		#mettre les premier percentage % dans train
		for i in rand_order1:
			for j in range(len(train)):
				self.train_data[j].append(train[j][i])
			self.train_labels.append(train_labels[i])
		
		#mettre les 100 - percentage*100 % dernier dans validation
		for i in rand_order1:
			for j in range(len(train)):
				self.validation_data[j].append(train[j][i])
			self.validation_labels.append(train_labels[i])
			
	
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


	def build_tree(self, train, train_labels, att_index, default, av_index, height):
		"""
		construire l'arbre de décision a l'aide d'appel récursifes

		train: l'ensemble d'exemples dans un array d'array
		train_labels: array des labels pour les exemples
		att_index: un array de l'index des attributs disponible
		av_index: les index qui sont encore disponnible
		av_ex_index: les index des liste d'exemples encore disponible (car en enlève des exemples)
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
		elif (av_att == 0) or (height == 0):
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
			new_av_index = av_index.copy()
			for i in range(len(att_index)):
				if att_index[i] == att_index[best_att_index]:
					new_av_index[i] = 0
			branches = []
					
			for p_value in best_att_freq: 
				# obtenir les arrays des exemples updater avec la valeur = de lattribut = p_value
				temp_train = []
				for i in range(len(att_index)):
					temp_train.append([])
				temp_labels = [] 
				for i in range(len(train[best_att_index])):
					if best_att_array[i] == p_value :
						for j in range(len(att_index)):
							temp_train[j].append(train[j][i])
						temp_labels.append(train_labels[i])
			
				#crée le sous arbre
				sub_tree = self.build_tree(temp_train, temp_labels, att_index, majority, new_av_index, height - 1)

				#ajouter les branches avec sub_tree
				branche = [p_value, sub_tree]
				branches.append(branche)
			
			tree.append(branches)
			
			return tree

	
	def build_tree_old(self, train, train_labels, att_index, default, av_index):
		"""
		construire l'arbre de décision a l'aide d'appel récursifes

		train: l'ensemble d'exemples dans un array d'array
		train_labels: array des labels pour les exemples
		att_index: un array de l'index des attributs disponible
		av_index: les index qui sont encore disponnible
		av_ex_index: les index des liste d'exemples encore disponible (car en enlève des exemples)
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
			new_av_index = av_index.copy()
			for i in range(len(att_index)):
				if att_index[i] == att_index[best_att_index]:
					new_av_index[i] = 0
			branches = []
					
			for p_value in best_att_freq: 
				# obtenir les arrays des exemples updater avec la valeur = de lattribut = p_value
				temp_train = []
				for i in range(len(att_index)):
					temp_train.append([])
				temp_labels = [] 
				for i in range(len(train[best_att_index])):
					if best_att_array[i] == p_value :
						for j in range(len(att_index)):
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
			attribute_data_array = attributes_data_matrice[attributes.index(attribute)]
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
			if fraction == 1.0:
				return gain
			else: 
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


	def reduire_nb_valeur(self, train, train_labels):
		"""
		Discrétise les valeurs continue dans l'ensemble train. retourne un nouveau train discretisé

		train: l'ensemble d'exemples dans un array d'array. ligne = exemple, colonne = attribut
		train_labels: array des labels pour les exemples
		"""
		reduced_train = []
		for i in range(len(train)):
			attribute_arr = train[i]
			attribute_max = [float("-inf")]
			#déterminer si l'index est a modifier
			att_type = type(train[i][0])
			if att_type == type(1) or att_type == type(1.0):
				freqs = self.get_frequencies(train[i])
				if len(freqs) > self.nb_val_max:
					#réduire les array au index
					attribute_max = max(attribute_arr)
					attribute_arr = self.reduire_array(attribute_arr, train_labels, attribute_max)
					
					
			reduced_train.append(attribute_arr)
			self.intervales[i].append([attribute_max])
		
		return reduced_train

	def reduire_array(self, attribut_array, train_labels, max_val):
		"""
		Discrétise les valeurs continue d'un array pour un attribut. Transforme ces valeurs continue en string représentant 
		des intervales de valerus continue. 
		
		note: self.nb_coupe est >= 1
		Si self.nb_coupe <= 1:  on utilise 1 point de coupe comme vue dans le cours.
		si self.nb_coupe > 1: on utilise des intervalles uniforme qui ne prennent pas en compte le gain d'information 

		attribut_array: liste contenant les valeurs continue de l'attribut dans le même ordre que train_labels
		train_labels: liste des labels.
		"""
		#cas 1 points de coupe
		if self.nb_coupe <= 1:

			#crée nouveau array de tuple
			combined = []
			for i in range(len(train_labels)):
				combined.append((attribut_array[i], train_labels[i]))

			#ordonné l'array
			combined.sort() #car val continue premier élément du tuple, peut utilise default sort

			#caculer les points de coupe et intervales
			pdc = [] #index des points de coupe
			intervales = [] #comme combined, mais avec les valeurs remplacer par intervalles
			for i in range(len(combined) - 1):
				if combined[i][1] != combined[i+1][1]:
					middle = (combined[i][0] + combined[i + 1][0]) /  2
					temp_att_data = []
					temp_labels_data = []
					itv_string = None
					itv_string1 = "]-infini,{}[".format(middle)
					itv_string2 = "[{}, infini[".format(middle)
					for j in range(len(combined)):
						if combined[j][0] < middle:
							itv_string = itv_string1
						else:
							itv_string = itv_string2

						temp_att_data.append(itv_string)
						temp_labels_data.append(combined[j][1])

					intervales.append(temp_att_data)
					intervales.append(temp_labels_data)
					pdc.append([middle, intervales, itv_string1, itv_string2])

			#trouver le meilleur points de coupe
			gains = []
			for i in range(len(pdc)):
				gain_ratio = self.get_gain_ratio(pdc[i][1][1], pdc[i][1][0])
				gains.append(gain_ratio)

			max_index = gains.index(max(gains))
			max_pdc = pdc[max_index]

			#transformer valeurs dans array
			return max_pdc[1][0]

		#cas range buckets
		else: 
			#déterminer la taille des intervales
			nb_intervales = self.nb_coupe + 1
			intervale_size = max_val / nb_intervales

			#déterminer les intervales
			start = intervale_size
			finish = intervale_size + intervale_size 
			intervale_names = []
			

			for i in range(self.nb_coupe):
				if i == 0:
					intervale_names.append(["]-infini, {}[".format(start), start])
				if i == (self.nb_coupe - 1):
					intervale_names.append(["[{}, infini[".format(start), float('inf')])
				else:
					intervale_names.append(["[{}, {}[".format(start, finish), finish])
					start += intervale_size
					finish += intervale_size

			#renomer selon les intervales
			renamed = []
			for i in range(len(train_labels)):
				itv_name = "oof"
				for j in range(len(intervale_names)):
					if attribut_array[i] < intervale_names[j][1]:
						itv_name = intervale_names[j][0]
						renamed.append(itv_name)

			return renamed


class NeuralNet(Classifier):
    
	def __init__(self, widht=10, max_iter=1000, learning_rate=0.1):
		self.widht: int = widht # nombre de neurones dans la couche cachée. 10 par défaut
		self.max_iter: int = max_iter # nombre d'itérations de descente de gradient. 1000 par défaut
		self.learning_rate: float = learning_rate # taux d'apprentissage. 0.1 par défaut

		self.trained: bool = False # booléen indiquant si le modèle a été entraîné
		self.depth: int = 1 # nombre de couches cachées. 1 pour le TP
		self.w1: np.array = None # array de poids entre la couche d'entrée et la couche cachée
		self.w2: np.array = None # array (de poids entre la couche cachée et la couche de sortie
		self.b1: np.array = None # vecteur de biais entre la couche d'entrée et la couche cachée
		self.b2: np.array = None # vecteur de biais entre la couche cachée et la couche de sortie
		self.classes: np.array = None # array des classes observées lors de l'entraînement

	def train(self, train, train_labels, early_stopping_rounds=10, tol=1e-4): 
		"""
		C'est la méthode qui va entrainer votre modèle,
		train est une matrice de type Numpy et de taille n*m, avec 
			n : le nombre d'exemple d'entrainement dans le dataset
			m : le nombre d'attributs (le nombre de caractéristiques)
		
		train_labels : est une matrice numpy de taille n*1
		
		early_stopping_rounds, tol : ces deux variables sont utilisés pour le critère d'arrêt précoce (tol=tolerance)
		
		"""
		self.trained = True
		self.classes = np.unique(train_labels)
  
		n, m = train.shape
		n_classes = len(np.unique(train_labels))

		# 1. Initialisation des poids et des biais
		self.w1 = np.random.randn(m, self.widht) * 0.01
		self.w2 = np.random.randn(self.widht, n_classes) * 0.01
		self.b1 = np.zeros((1, self.widht))
		self.b2 = np.zeros((1, n_classes))
		print(self.w1, self.w2, self.b1, self.b2)

		# Transformation des étiquettes en représentation one-hot
		one_hot_labels = np.eye(n_classes)[train_labels.reshape(-1)]

		best_loss = float('inf')
		no_improvement = 0

		# 2. Boucle d'entraînement
		for i in range(1, self.max_iter + 1):
			loss = 0
			dw1, dw2, db1, db2 = 0, 0, 0, 0

			for x, y_true in zip(train, one_hot_labels):

				# 2.a. Propagation avant (forward pass)
				_, (z1, a1, z2, a2) = self.predict(x)

				# 2.b. Calcul de l'erreur
				error = y_true - a2
				loss += -np.sum(y_true * np.log(a2 + 1e-9))

				# 2.c. Rétropropagation (backpropagation). "dx" représente la dérivée de l'erreur par rapport à "x".
				da2 = error
				dz2 = da2 * a2 * (1 - a2)
				dw2 += np.dot(a1.T, dz2)
				db2 += np.sum(dz2, axis=0, keepdims=True)
				da1 = np.dot(dz2, self.w2.T)
				dz1 = da1 * a1 * (1 - a1)
				dw1 += np.dot(x.reshape(-1, 1), dz1)
				db1 += np.sum(dz1, axis=0, keepdims=True)

			# Mise à jour des poids et des biais
			self.w1 += self.learning_rate * dw1 / n
			self.w2 += self.learning_rate * dw2 / n
			self.b1 += self.learning_rate * db1 / n
			self.b2 += self.learning_rate * db2 / n

			# Calcul de la perte moyenne
			loss /= n

			# Vérification de la convergence et condition d'arrêt anticipé
			if abs(best_loss - loss) < tol:
				no_improvement += 1
			else:
				no_improvement = 0
				best_loss = loss

			if no_improvement >= early_stopping_rounds:
				print(f"Arrêt anticipé après {i} itérations.")
				break

			# Affichage de la perte à chaque dizaine d'itérations
			if i % 10 == 0 or i == 1:
				print(f"Iteration {i}, Loss: {loss}")
  

	def predict(self, x: np.ndarray):
		"""
		Prédire la classe d'un exemple x donné en entrée
		x est un np.array de taille 1*m où m est le nombre de features (le nombre d'attributs)
		"""
		if not self.trained:
			raise Exception("Le modèle n'est pas entrainé. Vous devez d'abord appeler la méthode train.")
		z1: np.array = np.dot(x, self.w1) + self.b1 # somme pondérée biaisée de la couche cachée
		a1: np.array = self.sigmoid(z1) # activation de la couche cachée
		z2: np.array = np.dot(a1, self.w2) + self.b2 # somme pondérée biaisée de la couche cachée
		a2: np.array = self.sigmoid(z2) # activation de la couche de sortie
		return (self.classes[np.argmax(a2)], # argmax retourne l'indice de la valeur la plus grande, cet indice correspond à la classe prédite
          (z1,a1,z2,a2)) # on retourne aussi les valeurs de z1, a1, z2 et a2 pour le calcul du gradient et la rétropropagation

	def evaluate(self, X, y):
		"""
		c'est la méthode qui va evaluer votre modèle sur les données X
		l'argument X est une matrice de type Numpy et de taille nxm, avec 
		n : le nombre d'exemple de test dans le dataset
		m : le mobre d'attribus (le nombre de caractéristiques)
		
		y : est une matrice numpy de taille nx1
		
		vous pouvez rajouter d'autres arguments, il suffit juste de
		les expliquer en commentaire
		"""
		matrix_labels = np.unique(np.concatenate((self.classes, y)))
		return super().evaluate(X, y, matrix_labels)
    
	@staticmethod
	def sigmoid(z):
		"""
		Cette fonction calcule la sigmoide de z
  		"""
		return 1 / (1 + np.exp(-z))

""" import statistiques as stats
import load_datasets
wine_neural = NeuralNet(learning_rate=0.1, max_iter=1000, widht=10)
# Charger/lire les datasets
keys = ['train', 'train_labels', 'test', 'test_labels']
wine = dict(zip(keys, load_datasets.load_abalone_dataset(0.7)))
# Entrainez votre classifieur
wine_neural.train(wine['train'], wine['train_labels'])
print("WINE DATASET (binaire)")
print(wine_neural.w1, wine_neural.w2, wine_neural.b1, wine_neural.b2)
confusion_wine_neural = wine_neural.evaluate(wine['test'], wine['test_labels'])
stat.print_stats(confusion_wine_neural, binary=False) """


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

"""
#build_tree and predit
att1 = ['e', 'e', 'n', 'p', 'p', 'p', 'n', 'e', 'e', 'p', 'e', 'n', 'n', 'p'] #ciel
att2 = ['c', 'c', 'c', 't', 'f', 'f', 'f', 't', 'f', 't', 't', 't', 'c', 't'] #temp
att3 = ['e', 'e', 'e', 'e', 'n', 'n', 'n', 'e', 'n', 'n', 'n', 'e', 'n', 'e'] #humidite
att4 = ['w', 's', 'w', 'w', 'w', 's', 's', 'w', 'w', 'w', 's', 's', 'w', 's'] #vent
matrice = [att1, att2, att3, att4]
bells = [0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0] #jouerTennis
indexes = [0,1,2,3]
b = NaiveBayes()
x = b.build_tree(matrice, bells, indexes, bells[0], [1,1,1,1])
print(type(x))
y = type(x)
print(type(y))
print(type(x[1]))
test1 = ['n', 'c', 'e', 'w']
test2 = ['e', 'c', 'e', 'w']
test3 = ['e', 'c', 'n', 'w']
test4 = ['p', 'c', 'e', 'w']
test5 = ['p', 'c', 'e', 's']
b.tree = x
print(b.predict(test1))
print(b.predict(test2))
print(b.predict(test3))
print(b.predict(test4))
print(b.predict(test5))
"""

"""
#train
att1 = ['e', 'e', 'n', 'p', 'p', 'p', 'n', 'e', 'e', 'p', 'e', 'n', 'n', 'p'] #ciel
att2 = ['c', 'c', 'c', 't', 'f', 'f', 'f', 't', 'f', 't', 't', 't', 'c', 't'] #temp
att3 = ['e', 'e', 'e', 'e', 'n', 'n', 'n', 'e', 'n', 'n', 'n', 'e', 'n', 'e'] #humidite
att4 = ['w', 's', 'w', 'w', 'w', 's', 's', 'w', 'w', 'w', 's', 's', 'w', 's'] #vent
matrice = [att1, att2, att3, att4]
bells = [0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0] #jouerTennis
indexes = [0,1,2,3]
b = NaiveBayes()
b.train(matrice, bells, indexes,max(set(bells), key= bells.count))
"""

"""
#evaluate
att1 = ['e', 'e', 'n', 'p', 'p', 'p', 'n', 'e', 'e', 'p', 'e', 'n', 'n', 'p'] #ciel
att2 = ['c', 'c', 'c', 't', 'f', 'f', 'f', 't', 'f', 't', 't', 't', 'c', 't'] #temps
att3 = ['e', 'e', 'e', 'e', 'n', 'n', 'n', 'e', 'n', 'n', 'n', 'e', 'n', 'e'] #humidite
att4 = ['w', 's', 'w', 'w', 'w', 's', 's', 'w', 'w', 'w', 's', 's', 'w', 's'] #vent
matrice = np.array([att1, att2, att3, att4])
matrice = matrice.transpose()
bells = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0]) #jouerTennis
bells = bells.transpose()
indexes = [0,1,2,3]
b = ArbreDecision()
b.train(matrice, bells, indexes, 10, 10)
test = b.evaluate(b.validation_data, b.validation_labels) #['ciel', 'temps', 'humidité', "vent"])
stat.print_stats(test)
print("ok")
"""

"""
#reduire_array et reduire nb_valeur
arr1 = [40, 60, 90, 72, 48, 80]
arr2 = [0.25, 0.1, 0.8, 0.5, 0.7, 0.2]
arr3 = [0.3, 0.3, 0.2, 0.2, 0.3, 0.2]
matrice = [arr1, arr2, arr3]
bels = [0, 1, 0, 1, 0, 1]
b = ArbreDecision()
b.nb_coupe = 2
b.nb_val_max = 5
test = b.reduire_nb_valeur( matrice, bels)
print("ok")
"""

"""
#test finale pour hyper param
arr1 = [40, 60, 90, 72, 48, 80]
arr2 = [0.25, 0.1, 0.8, 0.5, 0.7, 0.2]
arr3 = [0.3, 0.3, 0.2, 0.2, 0.3, 0.2]
matrice = np.array([arr1, arr2, arr3])
matrice = matrice.transpose()
bels = np.array([0, 1, 0, 1, 0, 1])
bels = bels.transpose()
b = ArbreDecision()
indexes = [0,1,2]
b.train(matrice, bels, indexes, 10, 2, 5)
test = b.evaluate(b.validation_data, b.validation_labels)
stat.print_stats(test)
"""

