#Copyright: Fernando Cruz (and Sonia Carvalho)

import numpy as np
import matplotlib.pyplot as plt

from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.statistics.sigmoid_function import sigmoid_function


class LogisticRegression:
    """
    The LogisticRegression is a logistic model using the L2 regularization.
    This model solves the logistic regression problem using an adapted Gradient Descent technique
    Parameters
    ----------
    l2_penalty: float
        The L2 regularization parameter
    alpha: float
        The learning rate
    max_iter: int
        The maximum number of iterations
    Attributes
    ----------
    theta: np.array
        The model parameters, namely the coefficients of the logistic model.
        For example, sigmoid(x0 * theta[0] + x1 * theta[1] + ...)
    theta_zero: float
        The intercept of the logistic model
    """
    def __init__(self, l2_penalty: float = 1, alpha: float = 0.001, max_iter: int = 1000, use_adaptive_alpha: bool = False):
        """
        Parameters
        ----------
        l2_penalty: float
            The L2 regularization parameter
        alpha: float
            The learning rate
        max_iter: int
            The maximum number of iterations
        use_adaptive_alpha: bool
            Characterizes the type of model fitting
        """
        # parameters
        self.l2_penalty = l2_penalty
        self.alpha = alpha
        self.max_iter = max_iter
        self.use_adaptive_alpha = use_adaptive_alpha

        # attributes
        self.theta = None
        self.theta_zero = None
        self.cost_history = {}

    def fit(self, dataset: Dataset):
        """
        Fit the model to the dataset, according to the user's preference on regular or adaptive fit
        """
        if self.use_adaptive_alpha is True:
            self._adaptive_fit(dataset)
        elif self.use_adaptive_alpha is False:
            self._regular_fit(dataset)

    def _regular_fit(self, dataset: Dataset) -> 'LogisticRegression':
        """
        Fit the model to the dataset (regular fit)
        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to
        Returns
        -------
        self: LogisticRegression
            The fitted model
        """
        m, n = dataset.shape()

        # initialize the model parameters
        self.theta = np.zeros(n)
        self.theta_zero = 0

        # gradient descent
        for i in range(self.max_iter):
            # predicted y
            y_pred = np.dot(dataset.X, self.theta) + self.theta_zero

            # apply sigmoid function
            y_pred = sigmoid_function(y_pred)

            # compute the gradient using the learning rate
            gradient = (self.alpha * (1 / m)) * np.dot(y_pred - dataset.y, dataset.X)

            # compute the penalty
            penalization_term = self.alpha * (self.l2_penalty / m) * self.theta

            # update the model parameters
            self.theta = self.theta - gradient - penalization_term
            self.theta_zero = self.theta_zero - (self.alpha * (1 / m)) * np.sum(y_pred - dataset.y)

            #exercicio 6.1 - updating cost_history dictionary (key=iteration nr, value = cost)
            custo = self.cost(dataset)
            self.cost_history[i] = custo

            #exercicio 6.3 - confirmar
            if i > 0:
                if np.abs(self.cost_history[i - 1] - self.cost_history[i]) < 0.0001:
                    break
        return self

    def _adaptive_fit(self, dataset: Dataset) -> 'RidgeRegression':
        """
        Fit the model to the dataset (adaptive fit)

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to

        Returns
        -------
        self: RidgeRegression
            The fitted model
        """
        m, n = dataset.shape()

        # initialize the model parameters
        self.theta = np.zeros(n)
        self.theta_zero = 0

        # gradient descent
        for i in range(self.max_iter):
            # predicted y
            y_pred = np.dot(dataset.X, self.theta) + self.theta_zero

            # computing and updating the gradient with the learning rate
            gradient = (self.alpha * (1 / m)) * np.dot(y_pred - dataset.y, dataset.X)

            # computing the penalty
            penalization_term = self.alpha * (self.l2_penalty / m) * self.theta

            # updating the model parameters
            self.theta = self.theta - gradient - penalization_term
            self.theta_zero = self.theta_zero - (self.alpha * (1 / m)) * np.sum(y_pred - dataset.y)

            #updating cost_history dictionary (key=iteration nr, value = cost)
            custo = self.cost(dataset)
            self.cost_history[i] = custo

            #updating alpha value if cost doesnt change
            if i > 0:
                if np.abs(self.cost_history[i - 1] - self.cost_history[i]) < 1:
                    self.alpha = self.alpha / 2
        return self

    def predict(self, dataset: Dataset) -> np.array:
        """
        Predict the output of the dataset
        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the output of
        Returns
        -------
        predictions: np.array
            The predictions of the dataset
        """
        predictions = sigmoid_function(np.dot(dataset.X, self.theta) + self.theta_zero)

        # convert the predictions to 0 or 1 (binarization)
        mask = predictions >= 0.5
        predictions[mask] = 1
        predictions[~mask] = 0
        return predictions

    def score(self, dataset: Dataset) -> float:
        """
        Compute the Mean Square Error of the model on the dataset
        Parameters
        ----------
        dataset: Dataset
            The dataset to compute the MSE on
        Returns
        -------
        mse: float
            The Mean Square Error of the model
        """
        y_pred = self.predict(dataset)
        return accuracy(dataset.y, y_pred)

    def cost(self, dataset: Dataset) -> float:
        """
        Compute the cost function (J function) of the model on the dataset using L2 regularization
        Parameters
        ----------
        dataset: Dataset
            The dataset to compute the cost function on
        Returns
        -------
        cost: float
            The cost function of the model
        """
        predictions = sigmoid_function(np.dot(dataset.X, self.theta) + self.theta_zero)
        cost = (-dataset.y * np.log(predictions)) - ((1 - dataset.y) * np.log(1 - predictions))
        cost = np.sum(cost) / dataset.shape()[0]
        cost = cost + (self.l2_penalty * np.sum(self.theta ** 2) / (2 * dataset.shape()[0]))
        return cost

    #exercicio 6.2
    def line_plot(self):
        Xiteracoes = list(self.cost_history.keys())
        Ycost = list(self.cost_history.values())
        plt.plot(Xiteracoes, Ycost, '-')
        return plt.show()


if __name__ == '__main__':
    # import dataset
    from si.data.dataset import Dataset
    from si.model_selection.split import train_test_split

    # load and split the dataset
    dataset_ = Dataset.from_random(600, 100, 2)
    dataset_train, dataset_test = train_test_split(dataset_, test_size=0.2)

    # fit the model
    model = LogisticRegression(l2_penalty=1, alpha=0.001, max_iter=1000)
    model.fit(dataset_train)

    # compute the score
    score = model.score(dataset_test)
    print(f"Score: {score}")
