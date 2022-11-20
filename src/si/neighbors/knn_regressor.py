import numpy as np
from typing import Callable
from ..data.dataset import Dataset
from ..metrics.rmse import rmse
from ..statistics.euclidean_distance import euclidean_distance
from ..model_selection.split import train_test_split


class KNNRegressor:

    def __init__(self, k, distance: Callable = euclidean_distance):
        self.k = k
        self.distance = distance

        self.dataset = None


    def fit(self, dataset):
        self.dataset = dataset
        return self


    def _get_closest_labels_means(self, sample):
        distances = self.distance(sample, self.dataset.X)
        indexes = np.argsort(distances)[:self.k]
        indexes_values = self.dataset.y[indexes]
        return np.mean(indexes_values)


    def predict(self, dataset):
        return np.apply_along_axis(self._get_closest_labels_means, axis=1, arr=dataset.X)


    def score(self, dataset):
        y_pred = self.predict(dataset)
        return rmse(dataset.y, y_pred)

