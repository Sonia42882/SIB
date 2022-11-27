import numpy as np
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy

class StackingClassifier:

    def __init__(self, models, final_model):
        self.models = models
        self.final_model = final_model

    def fit(self, dataset):
        for model in self.models:
            model.fit(dataset)
        predictions = np.array([model.predict(dataset) for model in self.models])
       #T - transposta
        dataset_final = Dataset(predictions.T, dataset.y)
        self.final_model.fit(dataset_final)
        return self

    def predict(self, dataset):
        predictions = np.array([model.predict(dataset) for model in self.models])
        dataset_final = Dataset(predictions.T, dataset.y)
        return self.final_model.predict(dataset_final)

    def score(self, dataset):
        return accuracy(dataset.y, self.predict(dataset))

