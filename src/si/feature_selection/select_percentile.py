import numpy as np
from si.data.dataset import Dataset
from si.statistics.f_classification import f_classification

class SelectPercentile:
    def __init__(self, score_func, percentile):
        self.score_func = score_func
        self.percentile = percentile
        self.F = None
        self.p = None

    def fit(self, dataset):
        self.F, self.p = self.score_func(dataset)
        return self

    def transform(self, dataset):
        k = round(len(dataset.features) * self.percentile * 0.01) #CONFIRMAR
        idxs = np.argsort(self.F)[-k:]
        features = np.array(dataset.features)[idxs]
        return Dataset(X=dataset.X[:, idxs], y=dataset.y, features=list(features), label=dataset.label)

    def fit_transform(self, dataset):
        self.fit(dataset)
        return self.transform(dataset)


