
class Dense:

    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        #parâmetros da layer
        #cada layer recebe pesos que multiplica pelos dados de input e adiciona bias para evitar overfitting
        self.weights = np.randn(input_size, output_size) #so inicialização de matriz de pesos
        #CONFIRMAR: randon? randn? não é random (ver github do fernando)
        #self.weights contem todos os pesos da layer, vai ser sempre uma matriz
        self.bias = np.zeros((1, output_size)) #apenas inicialização, pode tomar todos os valores que quiser
        #adiciono um bias por cada output
        #npzeros continua a ser uma matriz que tem uma linha e tem output_size, neste caso tres

    def forward_propagation(self, input_data: np.array):
        return np.dot(input_data, self.weights) + self.bias
    #npdot por defeito faz multiplicação de matrizes
    #já demos em varias aulas para multiplicar vectores por matrizes
    #quando se passa um vector é que tem de se escolher axis em que se faz multiplicação

    #input_size tem ou não quer ser igual ao numero de features do dataset (colunas)?
    #sim, uma das regras de multiplicação de matrizes:
    #nr de colunas da segunda matriz igual ao nr de linhas da primeira

    #neste metodo, multiplico matriz de pesos pela matriz de input_data



class SigmoidActivation:

    def __init__(self):
        pass
        #função estática, não tem parâmetros, não calcula nada de forma diferente, igual em todos os casos
        #só tem de receber um input

    def forward(self, input_data): #retorna um np.array
        return 1/(1+ np.exp(-input_data))
        # para fazer ativação, assumindo que é uma layer













