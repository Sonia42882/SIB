import numpy as np
from typing import Callable
from ..data.dataset import Dataset
from ..metrics.rmse import rmse
from ..statistics.euclidean_distance import euclidean_distance
from ..model_selection.split import train_test_split


class KNNRegressor:
    """
     KNN Regressor
     KNN Regressor algorithm is similar to KNNClassifier, but in this case it should be applied to regression problems. This algorithm predicts a mean value of k more similar samples.
     Parameters
     ----------
     k: int
         The number of k examples to consider
     distance: Callable
         The distance function to use
     Attributes
     ----------
     dataset: np.ndarray
         The training data
     """
    def __init__(self, k: int, distance: Callable = euclidean_distance):
        """
        Initialize the KNNRegressor
        Parameters
        ----------
        k: int
            The number of k examples to consider
        distance: Callable
            The distance function to use
        """
        self.k = k
        self.distance = distance

        self.dataset = None


    def fit(self, dataset: Dataset):
        """
        It fits the model to the given dataset
        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to
        Returns
        -------
        self: KNNRegressor
            The fitted model
        """
        self.dataset = dataset
        return self


    def _get_closest_labels_means(self, sample: np.ndarray):
        """
        It returns the mean of the closest k examples of the given sample
        Parameters
        ----------
        sample: np.ndarray
            The sample to get the mean of the closest k examples
        Returns
        -------
        closest_labels_means: np.ndarray
            The mean of the closest k examples
        """
        distances = self.distance(sample, self.dataset.X)
        indexes = np.argsort(distances)[:self.k]
        indexes_values = self.dataset.y[indexes]
        return np.mean(indexes_values)


    def predict(self, dataset: Dataset):
        """
        It predicts the classes of the given dataset
        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the classes of
        Returns
        -------
        predictions: np.ndarray
            The predictions of the model
        """
        return np.apply_along_axis(self._get_closest_labels_means, axis=1, arr=dataset.X)


    def score(self, dataset: Dataset):
        """
        It returns the accuracy of the model on the given dataset
        Parameters
        ----------
        dataset: Dataset
            The dataset to evaluate the model on
        Returns
        -------
        accuracy: float
            The accuracy of the model
        """
        y_pred = self.predict(dataset)
        return rmse(dataset.y, y_pred)
