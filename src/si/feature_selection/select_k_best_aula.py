#não terminado
import numpy as np

class SelectBest:
    def __init__(self, score_func, k):
        self.score_func = score_func
        self.k = k
        self.F = None
        self.p = None

    def fit(self, dataset):
        self.F, self.p = self.score_func(dataset)
        return self

    def transform(self, dataset):
        #idxs = np.argsort(self.F)
        #retorna por ordem crescente os indexes do F
        #se o quinto valor fosse o mais baixo, no inicio retorna-me a posição quatro e assim adiante
        #de seguida, vou buscar as 10 melhores
        #como é crescente, vou buscar ao contrário - como faço isso? com menos
        #começar ao contrário
        #podemos fazer de outra maneira, o numpy só faz crescente
        #neste momento não faz mask, dá uma array de indexes
        #se o k for 10 dá os indices dos F mais novos
        #retorna um dataset novo
        #naquele caso fazia mask no caso em que variancia é diferene

#ver upload no github
#na 3.3 só usamos o iris
