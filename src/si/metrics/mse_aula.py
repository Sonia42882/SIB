import numpy as np


def rmse(y_true, y_pred):
    n = y_true.shape[0]  # confirmar
    rmse = np.sqrt((np.sum(y_true - y_pred) ^ 2) / n)
    return rmse

