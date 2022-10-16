import numpy as np


class Kmeans:
    def __init__(self, k, max_iter, distance):
        #parameters
        self.k = k
        self.max_iter = max_iter
        self.distance = distance

        #attributes, só são preenchidos durante o fit
        self.centroids = None
        self.labels = None

    def init_k_centroids(self):
        #centroides têm amostras dentro
        seeds = np.random.permutation(dataset.shape()[0])[:self.k]
        #só quero k centroides, vou buscar primeiros k posições das amostras
        self.centroids = dataset.X[seeds]
        #se k=2 aqueles centroides tem [0.1,0.3,...],[0.5,0.3,...]- tem dois vectores

    def closest_centroid(self, sample):
        centroids_distances = self.distance(sample, self.centroids)
        closest_centroid_index = np.argmin(centroids_distances, axis = 0)
        #index do vector com menor valor, axis=0 porque vector é em linhas
        return closest_centroid_index

    #np.apply_along_axis, axis=1
    #centroid = x[labels ==1]
    #centroid_mean = np.mean(centroid)
    #não preciso de decorar axis, posso experimentar no jupyter noteebok e altero se estiver errado
    #repetir até minimizar as distâncias (while loop até bater o nr máximo iterações ou não haver diferença nas labels)
    #fazer constantemente os passos 2 e 3

    convergence = False
    i = 0
