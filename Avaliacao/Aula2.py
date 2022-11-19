import numpy as np
from si.feature_selection.select_percentile import SelectPercentile
from si.data.dataset import Dataset
from si.io.csv import read_csv
from si.statistics.f_classification import f_classification

#Testar a class SelectPercentile usando o dataset iris
iris = read_csv("C:/Users/sonia/SIB/datasets/iris.csv", sep=',', features=True, label=True)
print(iris.shape())
percent = SelectPercentile(f_classification, 20)
percent.fit(iris)
transformed_iris = percent.transform(iris)
print(transformed_iris.shape())
