import load_datasets
import Classifiers # importer les classes du classifieur bayesien et Knn
#importer d'autres fichiers et classes si vous en avez développés
import statistiques as stat # importer les fonctions de calcul des métriques

"""
C'est le fichier main duquel nous allons tout lancer
Vous allez dire en commentaire c'est quoi les paramètres que vous avez utilisés
En gros, vous allez :
1- Initialiser votre classifieur avec ses paramètres
2- Charger les datasets
3- Entrainer votre classifieur
4- Le tester

"""

# Initialisez vos paramètres
TRAIN_RATIO = 0.7



# Initialisez/instanciez vos classifieurs avec leurs paramètres
iris_neural = Classifiers.NeuralNet(learning_rate=0.1, max_iter=1000, widht=10)
wine_neural = Classifiers.NeuralNet(learning_rate=0.1, max_iter=1000, widht=10)
abalone_neural = Classifiers.NeuralNet(learning_rate=0.1, max_iter=1000, widht=10)
iris_arbre = Classifiers.DecisionTree()
wine_arbre = Classifiers.DecisionTree()
abalone_arbre = Classifiers.DecisionTree()


# Charger/lire les datasets
keys = ['train', 'train_labels', 'test', 'test_labels']
iris = dict(zip(keys, load_datasets.load_iris_dataset(TRAIN_RATIO)))
wine = dict(zip(keys, load_datasets.load_wine_dataset(TRAIN_RATIO)))
abalone = dict(zip(keys, load_datasets.load_abalone_dataset(TRAIN_RATIO)))



# Entrainez votre classifieur
iris_neural.train(iris['train'], iris['train_labels'])
wine_neural.train(wine['train'], wine['train_labels'])
abalone_neural.train(abalone['train'], abalone['train_labels'])
    

iris_arbre.train(iris['train'], iris['train_labels'], list(range(0,len((iris['train'])[0]))), 30, 100000000, 20000000000)
wine_arbre.train(wine['train'], wine['train_labels'], list(range(0,len((wine['train'])[0]))), 30, 100000000, 20000000000)
#abalone_arbre.train(abalone['train'], abalone['train_labels'], list(range(0,len((abalone['train'])[0]))), 30, 100000000, 20000000000)

"""
Après avoir fait l'entrainement, évaluez votre modèle sur 
les données d'entrainement.
IMPORTANT : 
    Vous devez afficher ici avec la commande print() de python,
    - la matrice de confusion (confusion matrix)
    - l'accuracy
    - la précision (precision)
    - le rappel (recall)
    - le F1-score
"""
confusion_iris_neural = iris_neural.evaluate(iris['train'], iris['train_labels'])
confusion_wine_neural = wine_neural.evaluate(wine['train'], wine['train_labels'])
confusion_abalone_neural = abalone_neural.evaluate(abalone['train'], abalone['train_labels'])

print("\n█████████████ Réseau de neurones █████████████\n")
print("IRIS DATASET (multi-classes)")
stat.print_stats(confusion_iris_neural)

print("WINE DATASET (binaire)")
stat.print_stats(confusion_wine_neural, binary=True)

print("ABALONE DATASET (multi-classes)")
stat.print_stats(confusion_abalone_neural)




print("\n█████████████ Arbre de décision █████████████\n")
confusion_iris_arbre = iris_arbre.evaluate(iris['test'], iris['test_labels'])
confusion_wine_arbre = iris_arbre.evaluate(wine['test'], wine['test_labels'])
#confusion_abalone_arbre = iris_arbre.evaluate(iris['test'], iris['test_labels'])


print("IRIS DATASET (multi-classes)")
stat.print_stats(confusion_iris_arbre)
""
print("WINE DATASET (binaire)")
stat.print_stats(confusion_wine_arbre)

print("ABALONE DATASET (multi-classes)")
#stat.print_stats(confusion_abalone_arbre)

"""
Finalement, évaluez votre modèle sur les données de test.
IMPORTANT : 
    Vous devez afficher ici avec la commande print() de python,
    - la matrice de confusion (confusion matrix)
    - l'accuracy
    - la précision (precision)
    - le rappel (recall)
    - le F1-score
"""



#tests de base decision tree (la plus part non fonctionnel depuis les changements)

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
b = DecisionTree()
indexes = [0,1,2]
b.train(matrice, bels, indexes, 10, 2, 2)
temp_val1 = np.array(b.validation_data)
temp_val1 = temp_val1.transpose()
temp_val2 = np.array(b.validation_labels)
temp_val2 = temp_val2.transpose()
test = b.evaluate(temp_val1, temp_val2)
stat.print_stats(test)
"""




