
class NN:

    def __init__(self, layers):
        self.layers = layers #parâmetros
        #este modelo não tem atributos
        #vamos receber dados input, multiplicar, avançar, multiplicar, avançar

        #pode ser lista, as listas são ordenadas
        #podia ser um tuplo, tb são ordenados
        #a ordem interessa

    #qual a API que todos os modelos que definimos têm? FIT

    def fit(self, dataset): #retorna-se a ele próprio (NN)
        #fit treina o modelo com o dataset
        #iteramos cada camada e fazemos tais multiplicações que ja estão implementadas no forward
        x = dataset.X #o correto é criar aqui uma cópia para não alterar o dataset.X
        for layer in self.layers:
            x = layer.forward(x)
            #não poderia por layer.forward(dataset.X) senão estaria sempre a calcular nos dados iniciais
            #em python tudo são pointers, não cria uma cópia, sempre que estou a alterar o x, estou a alterar dataset.X


#slide SEIS
#quantas layers temos? duas layers
#de que tipo? dense layers, todos os nós totalmente conectados
#qual o input_size da primeira layers? dois (o +1 é o bias)
#qual o output_size da primeira layer? liga-se a dois nós na seguinte
#l1 = Dense(input_size = 2, output_size = ...)
#l2 = Dense (
#o output size da primeira layer, tem a segunda layer input size igual a essa
#tem que bater certo porque estamos a fazer multiplicação de matrizes
#de cada dense para cada dense temos uma sigmoid activation
#temos que criar as denses de sigmoid activation
#l1_sg = SigmoidActivation()
#l2_sg = SigmoidActivation()
#como defino agora o nosso modelo? o que meto? layers?
#nn_model = NN(layers(l1,l1_sg, l2, l2_sg))
#nn_model.fit(dataset)
#nn_model.predict #vemos na próxima aula
#apesar de ver nos exemplos que dense layers estao totalmente conectadas, têm sempre de sofrer ativação para passar para a proxima layer


#AVALIAÇÃO - talvez mais fácil fazer na próxima semana
#iris dataset é um problema de multiclasse, esta nova softmaxactivation é importante para este tipo de problema
#calcula a probibilidade de ocorência de cada uma das ocorrências
#enquanto a sigmoid só da zero e um e...

#ReLu ignora a parte negativa do x, dá os positivos

#os exercicios seguintes = o que fizemos na parte final (criar configuração com a tipologia do modelo) - jupyter, por ex
#ter 3 classes influencia ...
#é tudo crtC+V com algumas coisas a mudar, conforme tipo de problema
#ultimo deve acabar com uma ativação linear, nos não demos aqui nenhuma layer de activação linear
#considerem o que é um modelo linear, como implementar forward de todas as nossas layers

