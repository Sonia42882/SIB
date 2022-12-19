import numpy as np
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy

class StackingClassifier:
    """
    Ensemble classifier that uses a list of models to create predictions that are then used to train a final model.
    Parameters
    ----------
    models :  array-like, shape = [n_models]
        Different models for the ensemble
    final_model:
        Final model
    """

    def __init__(self, models, final_model):
        """
        Initialize the ensemble classifier.
        Parameters
        ----------
        models :  array-like, shape = [n_models]
            Different models for the ensemble
        final_model:
            Final model
        """
        self.models = models
        self.final_model = final_model

    def fit(self, dataset: Dataset):
        """
        Train the ensemble models.
        Parameters
        ----------
        dataset : Dataset
            The training data
        Returns
        -------
        self : StackingClassifier
            The fitted model.
        """
        for model in self.models:
            model.fit(dataset)
        predictions = np.array([model.predict(dataset) for model in self.models])
       #T - transposta
        dataset_final = Dataset(predictions.T, dataset.y)
        self.final_model.fit(dataset_final)
        return self

    def predict(self, dataset: Dataset):
        """
        Predict values for samples in X using trained models and the final model.
        Parameters
        ----------
        dataset : Dataset
            The test data.
        Returns
        -------
        predictions : np.ndarray
            The predicted values
        """
        predictions = np.array([model.predict(dataset) for model in self.models])
        dataset_final = Dataset(predictions.T, dataset.y)
        return self.final_model.predict(dataset_final)

    def score(self, dataset: Dataset):
        """
        Returns the error considering real values and predicted values.
        Parameters
        ----------
        dataset : Dataset
            The test data
        Returns
        -------
        score : float
            Mean accuracy
        """
        return accuracy(dataset.y, self.predict(dataset))

