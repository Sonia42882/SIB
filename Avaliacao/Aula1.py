import numpy as np
import pandas as pd
from src.si.io.csv import read_csv
from src.si.data.dataset import Dataset

#1.1
#carregar ficheiro iris
data = read_csv("C:/Users/sonia/SIB/datasets/iris.csv", sep=',', features=True, label=False)


#1.2
#selecciona a primeira variável independente
first_independent_variable = data.X[:,0]

#verifica a dimensão do array resultante
print(len(first_independent_variable))


#1.3
#seleciona as últimas 5 amostras do iris dataset
last_five = data.X[-5:]
print(last_five)

#qual a média das últimas 5 amostras para cada variável independente/feature
independent_variables_names = data.features[:-1]
#perceber quais são as variáveis independentes
print(independent_variables_names)
means = data.X[-5:,:-1].mean(axis=0)
#atenção - ultima variável não numérica, não se calcula média
#axis = 0 -> média por coluna, axis = 1 -> média por linha
print(means)


#1.4
#seleciona todas as amostras do dataset com valor superior ou igual a 1 para todas as features

####NÃO FUNCIONA ("TypeError: '>=' not supported between instances of 'str' and 'int'")
#arr = data.X
#unique_str_feature = np.unique(data.X[:,-1])
#print(unique_str_feature)
#mask = (arr >= int(1)) | (arr in unique_str_feature)
#new_arr = arr[mask]
#print("New array")
#print(new_arr)


#1.5
#seleciona todas as amostras com a classe/label igual a ‘Iris-setosa’
arr = data.X
mask_setosa = (data.X[:,-1]=="Iris-setosa")
new_arr = arr[mask_setosa]
print("New array - Iris-setosa")
print(new_arr)
#quantas amostras obténs?
print(len(new_arr))


#2 - tentativa para testar os métodos do exercicio2
data2 = read_csv("C:/Users/sonia/SIB/datasets/iris_missing_data.csv", sep=',', features=True, label=True)
#tamanho do dataset original
sh = data2.shape()
print(sh)
#retirar os NAs
remove_na = data2.dropna()
sa = data2.shape()
print(sa)

print("sonia")
data3 = read_csv("C:/Users/sonia/SIB/datasets/iris_missing_data.csv", sep=',', features=True, label=True)
print(data3.summary())
#substituir NA por valor
fn = data3.fillna(value = 100000)
print(data3.summary())

