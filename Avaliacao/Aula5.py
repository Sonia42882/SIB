from src.si.data.dataset import Dataset
from si.io.csv import read_csv
from si.linear_model.ridge_regression import RidgeRegression
from si.linear_model.logistic_regression import LogisticRegression
from si.model_selection.split import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


#exercicio 6.2
#testar para os dataset cpu
cpu_dataset = read_csv('../datasets/cpu.csv', features=True, label=True)
#standardization
cpu_dataset.X = StandardScaler().fit_transform(cpu_dataset.X)
#split dataset
train_dataset, test_dataset = train_test_split(cpu_dataset)
#logistic regression
rg = RidgeRegression(max_iter=2000)
rg.fit(train_dataset)
rg.predict(train_dataset)
rg.score(train_dataset)
rg.cost(train_dataset)
#graph
rg.line_plot()


#testar para os dataset breast
breast_dataset = read_csv('../datasets/breast-bin.csv', features=True, label=True)
#standardization
breast_dataset.X = StandardScaler().fit_transform(breast_dataset.X)
#split dataset
train_dataset, test_dataset = train_test_split(breast_dataset)
#logistic regression
lg = LogisticRegression(max_iter=2000)
lg.fit(train_dataset)
lg.predict(train_dataset)
lg.score(train_dataset)
lg.cost(train_dataset)
#graph
lg.line_plot()


#exercicio 6.4
#testar para os dataset cpu
cpu_dataset = read_csv('../datasets/cpu.csv', features=True, label=True)
#standardization
cpu_dataset.X = StandardScaler().fit_transform(cpu_dataset.X)
#split dataset
train_dataset, test_dataset = train_test_split(cpu_dataset)
#logistic regression
rg = RidgeRegression(l2_penalty= 1, alpha= 0.001, max_iter=2000, use_adaptive_alpha = True)
rg.fit(train_dataset)
rg.predict(train_dataset)
rg.score(train_dataset)
rg.cost(train_dataset)
#graph
rg.line_plot()


#testar para os dataset breast
breast_dataset = read_csv('../datasets/breast-bin.csv', features=True, label=True)
#standardization
breast_dataset.X = StandardScaler().fit_transform(breast_dataset.X)
#split dataset
train_dataset, test_dataset = train_test_split(breast_dataset)
#logistic regression
lg = LogisticRegression(max_iter=2000, use_adaptive_alpha = True)
lg.fit(train_dataset)
lg.predict(train_dataset)
lg.score(train_dataset)
lg.cost(train_dataset)
#graph
lg.line_plot()

