import numpy as np

def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray):
    """
    It returns the cross entropy error of the model on the given dataset

    Parameters
    ----------
    y_true: np.ndarray
        The true labels of the dataset
    y_pred: np.ndarray
        The predicted labels of the dataset
    """
    n = y_true.shape[0]
    #formula = - np.sum((y_true * np.log(y_pred))/ n
    return - np.sum(y_true * np.log(y_pred))/ n

def cross_entropy_derivative(y_true: np.ndarray, y_pred: np.ndarray):
    """
    It returns the derivative of the cross entropy function of the model on the given dataset

    Parameters
    ----------
    y_true: np.ndarray
        The true labels of the dataset
    y_pred: np.ndarray
        The predicted labels of the dataset
    """
    n = y_true.shape[0]
    formula = - (y_true / y_pred) + ((1 - y_true) / (1 - y_pred)) / n
    return formula
