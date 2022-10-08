# #contém o objeto Dataset
# import numpy as np
# import pandas as pd
#
# class Dataset:
#     def __init__(self, X, y, features, label):
#         #x é numpy array
#         #y é array de uma dimensão
#         #features pode ser uma lista de strings
#         #label é uma string (opcional)
#         self.X = X
#         self.y = y
#         self.features = features
#         self.label = label
#
#     def shape(self):
#         #tem que dar nr de exemplos do dataset e nr de features
#         #pode ser o shape do x e retornar
#         #retorna tuplo (primeiro valor = nr exemplos no dataset, segundo valor = nr de atributos)
#         return self.X.shape
#
#     def has_label(self):
#         #verifica se o dataset tem y
#         #retorna true ou false se y existe
#         if self.y is not None:
#             return True
#         else:
#             return False
#
#     def get_classes(self):
#         #lidar com erros, posso criar mensagens de alerta, diferente para supervisionado e não supervisionado
#         if self.y is None:
#             return None
#         return np.unique(self.y)
#
#     def get_mean(self):
#         return np.mean(self.X, axis=0)
#
#     def get_variance(self):
#         return np.var(self.X)
#
#     def get_median(self):
#         return np.median(self.X)
#
#     def get_min(self):
#         return np.min(self.X)
#
#     def get_max(self):
#         return np.max(self.X)
#
#     def summary(self):
#         #criar dataframe de raiz
#         #colunas f1, f2, f3 e linhas com media, mediana, etc
#         #df = pd.DataFrame(data, columns=['Name', 'Age'])
#
#         #columns = self.Y
#         data =
#
#         return pd.DataFrame(
#             {"mean": self.get_mean(),
#              "variance": self.get_variance(),
#             "median": self.get_median(),
#             "min": self.get_min(),
#             "max": self.get_max()}
#         )
#
#
#
#
# if __name__ == '__main__':
#     #não esta a funcionar
#     x = np.array([[1,2,3], [1,2,3]])
#     y = np.array([1,2])
#     features = ["A", "B", "C"]
#     label = "y"
#     #dataset = Dataset(X = x, y = y, features=features, label=label)
#     dataset = Dataset(X = x, y=None, features= features, label = None)
#     print(dataset.shape())
#     print(dataset.has_label())
#     print(dataset.get_classes())
#
#
#
#
#
#
