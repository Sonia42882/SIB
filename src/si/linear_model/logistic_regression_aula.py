from ..statistics.sigmoid_function_aula import sigmoid_function

class LogisticRegression:

    def __init__(self, l2_penalty, alpha, max_iter):
        self.l2_penalty = l2_penalty
        self.alpha = alpha
        self.max_iter = max_iter

        self.theta = None
        self.theta_zero = None


    def fit(self, dataset):
        # aplicar sigmoid_function é a diferença em relação ao logistic_regression
        m, n = dataset.shape()

        # initialize the model parameters
        self.theta = np.zeros(n)
        self.theta_zero = 0

        # gradient descent
        for i in range(self.max_iter):
            # predicted y
            y_pred = np.dot(dataset.X, self.theta) + self.theta_zero

            #apply sigmoid function -----> diferença em relação ao ridge_regression
            y_pred = sigmoid_function(y_pred)

            # computing and updating the gradient with the learning rate
            gradient = (self.alpha * (1 / m)) * np.dot(y_pred - dataset.y, dataset.X)

            # computing the penalty
            penalization_term = self.alpha * (self.l2_penalty / m) * self.theta

            # updating the model parameters
            self.theta = self.theta - gradient - penalization_term
            self.theta_zero = self.theta_zero - (self.alpha * (1 / m)) * np.sum(y_pred - dataset.y)

        return self

    def predict(self, dataset):
        #muda significativamente
        prediction = sigmoid_function(np.dot(dataset.X, self.theta) + self.theta_zero)
        mask = prediction #está incorreto
        prediction(mask) = 1
        prediction(mask) = 0
        return prediction

    def score(self, dataset): #igual, muda a accuracy
        y_pred = self.predict(dataset)
        return accuracy(dataset.y, y_pred)

    def cost(self, dataset):
        #não quero o binário aqui
        #o binario é só para ajustar a classificação
        #não concluido, ver github FC
        return

#AVALIAÇÃO
# pontos extra se opcional
# não sou penalizada se não tiver opcional
#
#
#
