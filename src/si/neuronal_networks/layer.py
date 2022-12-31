#copyright Fernando Cruz and Sónia Carvalho
import numpy as np


class Dense:
    """
    A dense layer is a layer where each neuron is connected to all neurons in the previous layer.
    Parameters
    ----------
    input_size: int
        The number of inputs the layer will receive.
    output_size: int
        The number of outputs the layer will produce.
    Attributes
    ----------
    weights: np.ndarray
        The weights of the layer.
    bias: np.ndarray
        The bias of the layer.
    """

    def __init__(self, input_size: int, output_size: int):
        """
        Initialize the dense layer.
        Parameters
        ----------
        input_size: int
            The number of inputs the layer will receive.
        output_size: int
            The number of outputs the layer will produce.
        """
        # parameters
        self.input_size = input_size
        self.output_size = output_size

        # attributes
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros((1, output_size))

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Performs a forward pass of the layer using the given input.
        Returns a 2d numpy array with shape (1, output_size).
        Parameters
        ----------
        X: np.ndarray
            The input to the layer.
        Returns
        -------
        output: np.ndarray
            The output of the layer.
        """
        return np.dot(X, self.weights) + self.bias

    def backward(self, error: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Performs a backward pass of the layer.
        Parameters
        ----------
        error: np.ndarray
            The error value of the loss function
        learning_rate: float
            Learning rate
        Returns
        -------
        output: np.ndarray
            The error of the previous layer.
        """
        error_to_propagate = np.dot(error, self.weights.T)
        self.weights -= learning_rate * np.dot(self.X.T, error)
        self.bias -= learning_rate * np.sum(error, axis=0)
        return error_to_propagate


class SigmoidActivation:
    """
    A sigmoid activation layer.
    """

    def __init__(self):
        """
        Initialize the sigmoid activation layer.
        """
        self.X = None


    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Performs a forward pass of the layer using the given input.
        Returns a 2d numpy array with shape (1, output_size).
        Parameters
        ----------
        X: np.ndarray
            The input to the layer.
        Returns
        -------
        output: np.ndarray
            The output of the layer.
        """
        self.X = X
        return 1 / (1 + np.exp(- self.X))

    def backward(self, error: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Performs a backward pass of the layer.
        Parameters
        ----------
        error: np.ndarray
            The error value of the loss function
        Returns
        -------
        output: np.ndarray
            The error of the previous layer.
        """
        deriv_sig = (1 / (1 + np.exp(- self.X))) * (1 - (1 / (1 + np.exp(- self.X))))
        error_to_propagate = error * deriv_sig
        return error_to_propagate



class SoftMaxActivation:
    """
    A SoftMax activation layer.
    """
    def __init__(self):
        """
        Initialize the SoftMax activation layer.
        """
        pass

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Performs a forward pass of the layer using the given input.
        Returns a 2d numpy array with shape (1, output_size).
        Parameters
        ----------
        X: np.ndarray
            The input to the layer.
        Returns
        -------
        output: np.ndarray
            The output of the layer.
        """
        ezi = np.exp(X - np.max(X))
        return ezi / (np.sum(ezi, axis=1, keepdims=True))


class ReLUActivation:
    """
    A rectified linear (ReLu) activation layer.
    """
    def __init__(self):
        """
        Initialize the ReLu activation layer.
        """
        self.X = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Performs a forward pass of the layer using the given input.
        Returns a 2d numpy array with shape (1, output_size).
        Parameters
        ----------
        X: np.ndarray
            The input to the layer.
        Returns
        -------
        output: np.ndarray
            The output of the layer.
        """
        self.X = X
        return  np.maximum(0, self.X)

    def backward(self, error: np.ndarray, learning_rate: float): #confirmar
        """
        Performs a backward pass of the layer.
        Parameters
        ----------
        X: np.ndarray
            The input data
        error: np.ndarray
            The error value of the loss function
        Returns
        -------
        output: np.ndarray
            The error of the previous layer.
        """
        relu_b = np.where(self.X > 0, 1, 0)
        error_to_propagate = error * relu_b
        return error_to_propagate


class LinearActivation:
    """
    A linear activation layer.
    """
    def __init__(self):
        """
        Initialize the linear activation layer.
        """
        pass

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Performs a forward pass of the layer using the given input.
        Returns a 2d numpy array with shape (1, output_size).
        Parameters
        ----------
        X: np.ndarray
            The input to the layer.
        Returns
        -------
        output: np.ndarray
            The output of the layer.
        """
        return  input_data
