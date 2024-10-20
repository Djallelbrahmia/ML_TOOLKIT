from abc import ABC, abstractmethod
import numpy as np

class Optimizer(ABC):
    """
    Abstract base class for all optimizers.
    """
    def __init__(self, learning_rate=0.01, batch_size=1):
        """
        Initializes the optimizer with a learning rate.
        :param learning_rate: Step size used for parameter updates
        """
        self.learning_rate = learning_rate
        self.batch_size = batch_size



    @abstractmethod
    def update(self, params, gradients):
        """
        Update parameters based on the gradients.
        This method should be implemented by all optimizers.
        
        :param params: List or array of model parameters
        :param gradients: List or array of gradients
        :return: Updated parameters
        """
        pass

