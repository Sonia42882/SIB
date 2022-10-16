import numpy as np


▪def euclidean_distance(x,y):
    """
• assinatura/argumentos:
- x – uma amostra, np array de uma dimensão
- y – várias amostras, np array de duas dimensões
• ouput esperado:
- array com distância entre X e as várias amostras de Y
"""
    return np.sqrt((x - y)**2)).sum(axis=1)



