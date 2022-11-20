#Podes testar o objeto KNNRegressor num jupyter notebook usando o dataset cpu.csv (regressão)
from si.io.csv import read_csv
from si.model_selection.split import train_test_split
from si.neighbors.knn_regressor import KNNRegressor

cpu_dataset = read_csv('../datasets/cpu.csv', features=True, label=True)

train_dataset, test_dataset = train_test_split(cpu_dataset)


knn = KNNRegressor(k = 2)
knn.fit(train_dataset)
predictions = knn.predict(test_dataset)
predictions
print("Predictions")
print(predictions)

print("Score")
score = knn.score(test_dataset)
print(score)
