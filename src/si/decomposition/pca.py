import numpy as np
from si.data.dataset import Dataset
from si.io.csv import read_csv


class PCA:
    """
    Performs principal component analysis (PCA) on the dataset, using SVD (Singular Value Decomposition) Linear Algebra method.
    Parameters
    ----------
    n_components: int
        Number of components to be computed

    Attributes
    ----------
    mean: np.ndarray
        Mean of the samples
    components: np.ndarray
        Principal components aka eigenvectors unitary matrix
    explained_variance: np.ndarray
        Explained variance aka eigenvalues' diagonal matrix.
    """

    def __init__(self, n_components: int):
        """
        PCA algorithm.
        Parameters
        ----------
        n_components: int
            Number of components to be computed
        """
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance = None

    def fit(self, dataset: Dataset):
        """
        Fits PCA on the dataset, estimating the mean, components and explained variance.
        Parameters
        ----------
        dataset: Dataset
            Dataset object.
        """
        #começa por centrar os dados
        self.mean = np.mean(dataset.X, axis=0)
        dados_centrados = dataset.X - self.mean

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

    def transform(self, dataset: Dataset):
        """
        Calculates the reduced dataset using SVD method.
        Parameters
        ----------
        dataset: Dataset
            Dataset object.
        Returns
        -------
        np.ndarray
            Transformed dataset.
        """
        #começa por centrar os dados
        self.mean = np.mean(dataset.X, axis=0)
        dados_centrados = dataset.X - self.mean

        #calcula o X reduzido - NÃO PERCEBI BEM
