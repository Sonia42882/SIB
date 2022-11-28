
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

#AULA 28.11
#aceita uma função que vai calcular, por ex
#soma dos quadrados dos erros ou outra
#ou podem adicionar metrica de erro por defeito (a mse)
#mais argumentos para o init?
#duvidas na lost function?
#fx em python são objetos

#def __init__(self, layers, epochs, learning_rate, loss, loss_derivate, verbose)
#atributos: self.history #guardo resultados (custo, erro ou outra coisa qualquer, diferença previstos / reais) para cada epoch

#o fit até agora fazia o backward propagation
#para cada layer, recebo na primeira o dataset de treino com features e exemplos
#passo à primeira layer, faz forward e no forward o que tinhamos implementado?
#substituimos o x para passar ao seguinte

#no backward propagation, espero que o forward acabe e depois? calculo o custo e alteramos os pesos anteriores
#nenhuma das nossas layers tem backward propagation
#fazemos isso quantas vezes? andar para frente, vejo o erro, ando para tras ---- nr de epochs que quero
#ciclo for com nr de epochs for epoch in range(1, self.epochs = 1)
#antes eu extraio o x e o y, mas só por conveniência, podia usar dataset.X e dataset.Y ao longo do algoritmo

#primeira iteraçao, primeiro passo é forward propagation (esta parte já tinhamos), o que fazemos a seguir?
#calculamos o custo e fazemos a backward propagation
#temos que usar a derivada da lost function
#para calcular o erro ou os pesos e atualizar e fazer backward propagation, tenho que fazer fx composta e derivar isso
#vamos lá na próxima aula
#depois vamos pegar no erro para cada exemplo e vamos passá-lo ao nosso backward

#esta diferença entre previstos e reais, usamos para atualizar os nossos pesos
#se usassemos so a loss, obtinhamos um float
#a nossa loss derivate vai dar para exemplo a diferença

#calculamos o erro, pego na ultima layer, invertemos a nossa lista de layers, pego na ultima e chamo o nosso backward erro e self.learning rate
#temos que ir a todas as layers e adicionar o nosso backward
#neste momento só diz para retornar o erro, porque vamos implementar isso na próxima aula

#por ultimo, o que fazemos? calculamos o verdadeiro custo a cada iteração
#mas daqui para baixo isto já nao importa para o treino, é só para perceber se estou a chegar ao minimom global nas primeiras 100 iterações
#depois posso otimização o nr de epochs se vir que o erro nao se atualiza ao fim de 300, por exemplo, e assim poupo recursos ao meu computador

#para aquela epoch guardo no meu dicionário - o que vai ter, por exemplo, na primeira iteração?
#vai ter 1 e um valor de 0,78, por exemplo

#por ultimo self.verbose e simplesmente faço um print para ter um nice output (isto é secundário)

#predict novamente

#esta avaliação fica para a próxima aula

