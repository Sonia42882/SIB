


class KMer:
    def __init__(self, k = 3):
        self.k = k #tamanho da substring
        self.k_mers = None #todos os k-mers possíveis
        #abordagem - criar dicionário, não iria resultar para todos os casos de estudo
        #nao ia compreender casos de teste com sequencias que não estavam no treino

        #posso pegar no alfabeto, calcular todas as combinações possíveis e verifico se a sequência está dentro de
        #todas as combinações possíveis
        #temos de ter todas as combinações possiveis, se não existirem damos zero

    def fit(self, dataset): #estima todos os k_mers possíveis; retorna o self (ele próprio)
        #como fazer todas as combinações possíveis? produto cartesiano, itertools tem fx product
        #iter true itertools.product (array alfabeto, parametro quantas xs quero repetir o array)
        #REPEAT
        #o que faço com isto?
        #itertools vai gerar todas as combinações possíveis
        #pode ser mais fácil fazer numa lista, não precisamos de guardar as contagens num dicionário
        #adiciono todas as combinações possíveis numa lista, onde faço isso? no fit? transform?
        #pode ser logo no init, o fernando no codigo fez no fit, poupa trabalho meter no fit

        # list(itertools.product([A, C, T, G], repeat = self.k)))
        #cada vez que itero vamos ter tuplo das combinações
        #infere todas as comninações possiveis de nucleotidos consoante o k
        #self.k_mers = lista de tuplos com ACG, ATG (porque k=3)


    def transform(self): #calcula a frequência normalizada de cada k mer em cada sequência
        #divide-se quantas vezes aparece o k_mer na sequencia pelo tamanho da sequência
        #agora crio um dicionário com todas as combinações possiveis que inferi e atribuo contagem zero
        #à medida que percorrro seq, atualizo a contagem do dicionario
        #dicio = {}
        #i = 0, adiciono k=3 e retiro a primeira janela (parecido com o blast simplificado)
        #atenção, quando chegar ao fim, tenho que iterar sempre para ter k=3
        counts = {k_mer:0 for k_mer in self.k_mers}
        for i in range(len(seq) - self.k +1):
            k_mer = sequence[i:i + self.k]
            #k_mer é string, k_mers tem tuplos
            #posso fazer tuplo de k_mer e criar dicionario de tuplos
            #itertools retorna lista de tuplos com strings, se fizer join ele itera o tuplo e adiciona-o à string
            #o fernando fez self.k_mers = ["".join ....]
            counts[k_mer] += 1
        return np.array([counts[k_mer] / len(sequence) for k_mer in self.k_mers])
            #separem a lógica em funções para se ler melhor, não façam 3 for loops seguidos


    def fit_transform(self): #corre o fit e depois o transform
        #
        #return

#AVALIAÇÃO
#o kmer que fizemos é puramente especifico para nucleotidos
#se desse seq de proteinas, nao conseguiria calcular nada
#quero que altero para que possa calcular a composição de sequencia nucleotidica e peptidica
#adaptar a esta classe que desenvolvemos na aula
#envolve alterar 2/3 linhas de codigo, mas temos que compreender bem o que é o kmer e o que temos que alterar
#adicionar um jupyter notebook ou script em que testo
#fazer protocolo que mostrou, kmer, gerar dataset com variaveis numéricas e usar standard scaler
#e normal nesse dataset ter piores resultados do que o que ele mostrou
#nao brinquem com k superiores a 3 - porque?
#vai demorar imenso tempo
