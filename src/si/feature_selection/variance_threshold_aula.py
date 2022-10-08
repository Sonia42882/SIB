# #não completo
#
# class VarianceThreshold: #retorna um novo dataset onde vamos aplicar a "magia", onde neste caso seleccionamos colunas que interessam
#     def __init__(self, threshold):
#         #quais os atributos da classe?
#         self.threshold = threshold
#         self.variance = None #para já vazio porque dataset está vazio
#
#     def fit(dataset):
#         #estima / calcula a variância de cada feature; retorna o self
#         #o nosso dataset é uma estrutura nossa, não é um numpy array
#        # variance = np.var(dataset.X)
#         variance = Dataset.get_var() #método da outra aula, é uma alternativa)
#         self.variance = variance
#         return self #retorna-se a ele próprio, assim dá para fazer códigos numa linha
#     #é uma API do transformer, ignorem por agora os motivos
#
#     def transform(dataset):
#         #selecciona todas as features com variância superior ao threshold e retorna o X seleccionado
#         mask = self.vaiance > self.threshold
#         novoX = dataset.X[:,mask]
#         #podemos fazer selecção com booleanos
#         #selecciona linhas por defeito novoX = dataset.X[mask]
#         #portanto tenho que por novoX = dataset.X[:,mask] - a virgula e ":"
#         #obrigatorio retornar o novo dataset
#         return Dataset(novoX, dataset.y, features)
#     #y antigo, features é uma lista de strings
#     #com lista não posso fazer seleção de masks
#     #então crio um array com a lista e seleciono outra vez com a mask
#     #volto a passar então uma lista
#
#
#
# if__name__ =="__main__":
# from si.data.dataset_aula import Dataset
#
# dataset = Dataset.from_random(100,10,2)
# dataset.X[:, 0] = 0
#
# dataset = Dataset(X = np.array([[0.2.0.3],[0,1,4,3],[0,1,1,3]])), y = np.array #faltam coisas
#
# selector = VarianceThreshold(theshold = 0.1)
# selector = selector.fit(dataset)
# new_dataset = selector.transfom(dataset)
# print(dataset.features)
#
#






