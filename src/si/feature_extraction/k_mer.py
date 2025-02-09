#copyright: Fernando Cruz and Sónia Carvalho

import itertools

import numpy as np

from si.data.dataset import Dataset


class KMer:
    """
    A sequence descriptor that returns the k-mer composition of the sequence.
    Parameters
    ----------
    k : int
        The k-mer length
    sequence: str
        The type of biological sequence (e.g., DNA or AA)
    Attributes
    ----------
    alphabet: str
        The alphabet of the biological sequence
    k_mers : list of str
        The k-mers.
    """
    def __init__(self, k: int = 2, sequence: str = "DNA"):
        """
        Parameters
        ----------
        k : int
            The k-mer length
        sequence: str
            The type of biological sequence (e.g., DNA or AA)
        """
        # parameters
        self.k = k
        self.sequence = sequence

        if self.sequence == "DNA":
            self.alphabet = "ACTG"
        elif self.sequence == "AA":
            self.alphabet = "ACDEFGHIKLMNPQRSTVWYXBZJ"

        # attributes
        self.k_mers = None

    def fit(self, dataset: Dataset) -> 'KMer':
        """
        Fits the descriptor to the dataset.
        Parameters
        ----------
        dataset : Dataset
            The dataset to fit the descriptor to.
        Returns
        -------
        KMer
            The fitted descriptor.
        """

        # generate the k-mers
        self.k_mers = [''.join(k_mer) for k_mer in itertools.product(self.alphabet, repeat=self.k)]
        return self

    def _get_sequence_k_mer_composition(self, sequence: str) -> np.ndarray:
        """
        Calculates the k-mer composition of the sequence.
        Parameters
        ----------
        sequence : str
            The sequence to calculate the k-mer composition for.
        Returns
        -------
        list of float
            The k-mer composition of the sequence.
        """
        # calculate the k-mer composition
        counts = {k_mer: 0 for k_mer in self.k_mers}

        for i in range(len(sequence) - self.k + 1):
            k_mer = sequence[i:i + self.k]
            counts[k_mer] += 1

        # normalize the counts
        return np.array([counts[k_mer] / len(sequence) for k_mer in self.k_mers])

    def transform(self, dataset: Dataset) -> Dataset:
        """
        Transforms the dataset.
        Parameters
        ----------
        dataset : Dataset
            The dataset to transform.
        Returns
        -------
        Dataset
            The transformed dataset.
        """
        # calculate the k-mer composition
        sequences_k_mer_composition = [self._get_sequence_k_mer_composition(sequence)
                                       for sequence in dataset.X[:, 0]]
        sequences_k_mer_composition = np.array(sequences_k_mer_composition)

        # create a new dataset
        return Dataset(X=sequences_k_mer_composition, y=dataset.y, features=self.k_mers, label=dataset.label)

    def fit_transform(self, dataset: Dataset) -> Dataset:
        """
        Fits the descriptor to the dataset and transforms the dataset.
        Parameters
        ----------
        dataset : Dataset
            The dataset to fit the descriptor to and transform.
        Returns
        -------
        Dataset
            The transformed dataset.
        """
        return self.fit(dataset).transform(dataset)


if __name__ == '__main__':
    from si.data.dataset import Dataset

    dataset_ = Dataset(X=np.array([['ACTGTTTAGCGGA', 'ACTGTTTAGCGGA']]),
                       y=np.array([1, 0]),
                       features=['sequence'],
                       label='label')

    k_mer_ = KMer(k=2)
    dataset_ = k_mer_.fit_transform(dataset_)
    print(dataset_.X)
    print(dataset_.features)
