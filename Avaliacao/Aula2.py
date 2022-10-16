import numpy as np
from si.feature_selection.select_percentile import SelectPercentile
from si.data.dataset import Dataset
from si.io.csv import read_csv
from si.statistics.f_classification import f_classification

#Testar a class SelectPercentile usando o dataset cpu.csv
data = read_csv("C:/Users/sonia/SIB/datasets/cpu.csv", sep=',', features=True, label=True)
print(data.shape())
percent = SelectPercentile(f_classification, 10)
sol = percent.fit_transform(data)
#print(sol.X)
print(sol.shape())
