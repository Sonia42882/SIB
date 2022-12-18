import numpy as np
from si.data.dataset import Dataset
from si.statistics.f_classification import f_classification

from typing import Callable

class SelectPercentile:
    """
    Select features according to the percentile chosen by the user.
    Feature ranking is performed by computing the scores of each feature using a scoring function:
        - f_classification: ANOVA F-value between label/feature for classification tasks.
        - f_regression: F-value obtained from F-value of r's pearson correlation coefficients for regression tasks.

    Parameters
    ----------
    score_func: callable
        Function taking dataset and returning a pair of arrays (scores, p_values)
    percentile: float
        Percentile to select the features
    Attributes
    ----------
    F: array, shape (n_features,)
        F scores of features.
    p: array, shape (n_features,)
        p-values of F-scores.
    """

    def __init__(self, score_func: Callable = f_classification, percentile: float):
        """
        Select features according to a percentile chosen by the user.
        Parameters
        ----------
        score_func: callable
            Function taking dataset and returning a pair of arrays (scores, p_values)
        percentile: float
            Percentile to select the features.
        """
        self.score_func = score_func
        self.percentile = percentile
        self.F = None
        self.p = None

    def fit(self, dataset: Dataset):
        """
        It fits SelectPercentile to compute the F scores and p-values.
        Parameters
        ----------
        dataset: Dataset
            A labeled dataset
        Returns
        -------
        self: object
            Returns self.
        """
        self.F, self.p = self.score_func(dataset)
        return self

    def transform(self, dataset: Dataset):
        """
        It transforms the dataset by selecting the number of features according to a percentile chosen by the user.
        Parameters
        ----------
        dataset: Dataset
            A labeled dataset
        Returns
        -------
        dataset: Dataset
            A labeled dataset with the number of features selected according to a percentile chosen by the user.
        """
        k = round(len(dataset.features) * self.percentile * 0.01) #CONFIRMAR
        idxs = np.argsort(self.F)[-k:]
        features = np.array(dataset.features)[idxs]
        return Dataset(X=dataset.X[:, idxs], y=dataset.y, features=list(features), label=dataset.label)

    def fit_transform(self, dataset: Dataset):
        """
        It fits SelectPercentile and transforms the dataset by selecting the number of features according to a percentile chosen by the user.
        Parameters
        ----------
        dataset: Dataset
            A labeled dataset
        Returns
        -------
        dataset: Dataset
            A labeled dataset with the number of features selected according to a percentile chosen by the user.
        """
        self.fit(dataset)
        return self.transform(dataset)


