#não completo
import numpy as np
from scipy import stats
#from - falta um


def f_classification(dataset):
    #o x e o y nesse caso têm de ser agrupados pelas labels
    # este x e o y do ml é consoante a label
    #em estatistica, variavel dependente e independente
    classes = dataset.get_classes()
    groups = [dataset.X[dataset.y==c] for c in classes]
    F, p = stats.f_oneway(*groups)
    return F, p
    # consoante as labels do nosso dataset
#uma variavel em estatistica é o conjunto de samples com a mesma label
#a minha variavel X0 em estatistica é igual a todas as amostras que têm, por exemplo, a classe1
# x0 seria 0 e a 2
#x1 seria 1 e 3
#porque x = [0 1 2 3] e y = [1 0 1 0]
#não percebi
#oneway aceita x0, x1, x2 e testa a variancia de todos uns com os outros
#o que o asterisco faz, se tenho isto numa lista [x0,x1,x2] ele passa para outro formado
#e como se estivesse tudo zipado numa lista e ele passa para para f one way(xx0,x1,x2)


def 






