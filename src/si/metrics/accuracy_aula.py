import numpy as np

def accuracy(y_true, y_pred):
    #conto os VP, VN e o tamanho do dataset
    return np.sum(y_true == y_pred) / len(y_true)

