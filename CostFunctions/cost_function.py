from abc import ABC, abstractmethod

class CostFunction(ABC):
    """
    Abstract base class for all cost functions.
    """

    @abstractmethod
    def compute(self, predictions, y):
        """
        Compute the cost given predictions and true values.

        :param predictions: Predicted values
        :param y: True labels
        :return: Cost value
        """
        pass


    def compute_gradients(self, predictions, targets):
        """
        Compute the gradients of the cost function with respect to the predictions.
        
        :param predictions: Predicted values from the model
        :param targets: True target values
        :return: Gradients with respect to the predictions
        """
        pass