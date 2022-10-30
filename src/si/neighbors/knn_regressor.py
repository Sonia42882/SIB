import numpy as np
from ..data.dataset import Dataset
from ..metrics.mse_aula import rmse
from ..statistics.euclidean_distance import euclidean_distance
from ..model_selection.split import train_test_split


class KNNClassifier:

    def __init__(self, k, distance):
        self.k = k
        self.distance = distance

        self.dataset = None


    def fit(self, dataset):
        self.dataset = dataset
        return self


    def _get_closest_label_r(self):
        distances = self.distance(sample, self.dataset.X)
        indexes = np.argsort(distances)[:self.k]
        indexes_values = self.dataset.y[k_nearest_neighbors]
        return np.mean(indexes_values)


    def predict(self):
        return np.apply_along_axis(self._get_closest_label_r, axis=1, arr=dataset.X)


    def score(self, dataset):
        y_pred = self.predict(dataset)
        return rmse(dataset.y, y_pred)

