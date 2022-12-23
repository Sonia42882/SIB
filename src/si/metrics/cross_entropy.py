import numpy as np

def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray): #confirmar
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
    formula = - np.sum((y_true * np.log(y_pred))/ n
    return
#FALTA DERIVAR
