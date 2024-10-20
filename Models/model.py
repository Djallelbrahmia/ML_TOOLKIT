from abc import ABC, abstractmethod

class Model(ABC):
    """
    Abstract base class for all models.
    """
    def __init__(self):
        self.params = None  # To hold model parameters

    @abstractmethod
    def predict(self, X):
        """
        Make predictions based on input data.
        :param X: Input features
        :return: Predicted values
        """
        pass

    @abstractmethod
    def compute_gradients(self, X, y):
        """
        Compute gradients for model parameters.
        :param X: Input features
        :param y: True labels
        :return: Gradients for the model parameters
        """
        pass

    @abstractmethod
    def fit(self, X, y, optimizer, num_epochs):
        """
        Fit the model to the training data.
        :param X: Input features
        :param y: True labels
        :param optimizer: Optimizer to use for training
        :param num_epochs: Number of epochs for training
        """
        pass
