from si.io.csv import read_csv
from si.model_selection.split import train_test_split
from si.neighbors.knn_classifier import KNNClassifier
from si.linear_model.logistic_regression import LogisticRegression
from si.ensemble.stacking_classifier import StackingClassifier
from si.ensemble.voting_classifier import VotingClassifier
from sklearn.preprocessing import StandardScaler

#exercicio dos slides
#Usa o dataset breast bin.csv
breast_dataset = read_csv('../datasets/breast-bin.csv', features=True, label=True)
#Usa o sklearn.preprocessing.StandardScaler para standardizar
breast_dataset.X = StandardScaler().fit_transform(breast_dataset.X)
#Divide o dataset em treino e teste
train_dataset, test_dataset = train_test_split(breast_dataset)
#Cria o modelo KNNClassifier
knn = KNNClassifier()
#Cria o modelo LogisticRegression
lg = LogisticRegression(max_iter=2000)
#Cria o modelo ensemble VotingClassifier usando os classificadores anteriores
voting = VotingClassifier([knn, lg])
#Treina o modelo. Qual o score obtido?
voting.fit(train_dataset)
score = voting.score(test_dataset)
print(score) #0.9784172661870504


#exercicio 6.2
#Usa o dataset breast bin.csv
breast_dataset = read_csv('../datasets/breast-bin.csv', features=True, label=True)
#Usa o sklearn.preprocessing.StandardScaler para standardizar
breast_dataset.X = StandardScaler().fit_transform(breast_dataset.X)
#Divide o dataset em treino e teste
train_dataset, test_dataset = train_test_split(breast_dataset)
#Cria o modelo KNNClassifier
knn = KNNClassifier()
#Cria o modelo LogisticRegression
lg = LogisticRegression(max_iter=2000)
#Cria um segundo modelo KNNClassifier
knn_final = KNNClassifier()
#Cria o modelo StackingClassifier usando os classificadores anteriores
stacking = StackingClassifier([knn, lg], knn_final)
#Treina o modelo. Qual o score obtido?
stacking.fit(train_dataset)
score = stacking.score(test_dataset)
print(score) #0.9784172661870504
