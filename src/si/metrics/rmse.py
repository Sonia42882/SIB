import numpy as np

def rmse(y_true: np.ndarray, y_pred: np.ndarray):
    """
    It returns the root mean squared error of the model on the given dataset
    Parameters
    ----------
    y_true: np.ndarray
        The true labels of the dataset
    y_pred: np.ndarray
        The predicted labels of the dataset
    Returns
    -------
    rmse: float
        The root mean squared error of the model
    """
    return np.sqrt(np.sum((y_true-y_pred)**2) / len(y_true))
