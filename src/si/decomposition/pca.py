import numpy as np
from si.data.dataset import Dataset
from si.io.csv import read_csv


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance = None

    def fit(self):
        #começa por centrar os dados
        self.mean = np.mean(dataset.X, axis=0)
        dados_centrados = dataset.X - self.mean #para que serve? atualizar com self?

        #calcula o SVD de X
        U, S, VT = numpy.linalg.svd(X, full_matrices=False)
        SVD = U * S * VT

        #infere os componentes principais
        self.components = VT[:self.n_components]

        #infere a variância explicada
        n = dataset.shape()[0]
        EV = (SVD ** 2) / (n - 1)
        self.explained_variance = EV[:self.n_components]
        return self

    def transform(self):
        #começa por centrar os dados
        self.mean = np.mean(dataset.X, axis=0)
        dados_centrados = dataset.X - self.mean

        #calcula o X reduzido - NÃO PERCEBI
