import numpy as np
from si.statistics.euclidean_distance import euclidean_distance


class KNNClassifier:
    def __init__(self, k, distance):
        self.k = k
        self.distance = distance
        sef.dataset = None

    def fit(self, dataset):
        self.dataset = dataset
        return self

    def predict(self, dataset):
        #escrever formula de distancia?
        distances = self.distance(sample, self.dataset.X)
        k_nearest_neighbors = np.argsort(distance)[:self.k]
        k_nearest_neighbors_labels = self.dataset.y[k_nearest_neighbors]
        labels, counts = np.unique(k_nearest_neighbors_labels, return_counts = True)
        #return_counts para retornar segundo array
        return labels[np.argmax(counts)]
        #argsort pega em array, ordena-o por ordem crescente e dá indexes desses valores
        #k exemplos mais semelhantes
        #como escolho a classe mais comum?
        # np.unique() - retorna array com contagens e valores unicos

    def get_closest_label(self, sample):
        distances = self.distance(sample, self.dataset.X)
        k_nearest_neighbors = np.argsort(distance)[:self.k]
        k_nearest_neighbors_labels = self.dataset.y[k_nearest_neighbors]
        labels, counts = np.unique(k_nearest_neighbors_labels, return_counts = True)
        return labels[np.argmax(counts)]

    def predict(self, dataset):
        return np.apply_along_axis(self._get_closest_label, axis = 1, arr = dataset.X)

    def score(self,dataset):
        #dataset_teste vai ter um y que têm labels, vai ser o true
        #chamo o predict neste dataset
        predictions = self.predict(dataset)
        return accuracy(dataset.y, predictions)


#avaliação - semelhantes mas para regressão (e não classificação)
#accuracy é apenas para classes
#distancia pode ser euclidean distance (dá para classificação e regressão)
#iris é classificação, cpu é regressão
#no ponto4, calculava a classe mais comum, aqui vamos usar a média, vamos ter valores continuos
#
