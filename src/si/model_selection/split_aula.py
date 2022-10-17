import numpy as np


def train_test_split(dataset, test_size, random_state):
    #mais facil definir test_size como fração
    np.random.seed(random_state)
    #sempre que corro amostra, tenho os mesmos, com seed consigo reproduzir pipeline c/ mesmo resultado
    n_samples = dataset.shape()[0]
    n_test = int(n_samples * test_size)
    permutation = np.random.permutation(n_samples)
    test_idxs = permutation[:n_test]
    train_idxs = permutation[n_test:]
    numtest = int(tamanho_dataset * test_size)
    train = Dataset(dataset.X[train_idxs], dataset.y[train_idxs], features = dataset.features, label = dataset.label)
    test =  Dataset(dataset.X[test_idxs], dataset.y[test_idxs], features = dataset.features, label = dataset.label)
    return





