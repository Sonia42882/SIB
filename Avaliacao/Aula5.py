#incorporar no restante código, apenas codigo com algumas ideias
#6.1
self.cost_history = {}

#depois do gradiente descent
custo = self.cost(dataset)
self.cost_history[i] = custo


#6.2 (grafico)
def line_plot(self):

    Xiteracoes = list(self.cost_history.keys())
    Ycost = list(self.cost_history.values())
    plt.plot(Xiteracoes, Ycost, '-')

    return plt.show()
#testar para os datasets cpu e breast

#6.3
if np.abs(self.cost_history.get(i - 1) - custo) >= 1: #alterar valor para logistic regression
    self.history[i] = custo
else:
    break
#testar para os datasets cpu e breast

#6.4 (opcional)
