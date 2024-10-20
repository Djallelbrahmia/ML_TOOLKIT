import numpy as np
from .cost_function import CostFunction

class MeanSquaredError(CostFunction):
    def compute(self, predictions, y):
        """
        Compute Mean Squared Error.

        :param predictions: Predicted values
        :param y: True labels
        :return: Mean Squared Error
        """
        return np.mean((predictions - y) ** 2)

    def compute_gradients(self, predictions, targets,X,model,activation_function=None):
        """
        Compute the gradients of the Mean Squared Error.
        
        :param predictions: Predicted values from the model
        :param targets: True target values
        :param model: model to return gradient related to that model 
        :param X: Input data

        :return: Gradients of the MSE
        """
        match model:
            case "LR" :
                n = len(targets)  # Number of samples
                errors = predictions - targets  # Error term

                # Gradient for weights A
                gradient_A = (2 / n) * np.dot(X.T, errors)  # Gradient w.r.t. weights A

                # Gradient for bias B
                gradient_B = (2 / n) * np.sum(errors)  # Gradient w.r.t. bias B

                return gradient_A, gradient_B
            case "NLR" :
                n = len(targets)  # Number of samples
                errors = predictions - targets  # Error term

                # Gradient for weights A
                gradient_A = (2 / n) * np.dot(X.T, errors)  # Gradient w.r.t. weights A

                # Gradient for bias B
                gradient_B = (2 / n) * np.sum(errors)  # Gradient w.r.t. bias B

                return gradient_A, gradient_B
            case "LC": 
                n = len(targets)
                errors = predictions - targets

                if activation_function == "sigmoid":
                    # Derivative of the sigmoid function
                    sigmoid_derivative = predictions * (1 - predictions)
                    errors *= sigmoid_derivative

                    # Gradient for weights (A)
                    gradient_A = (2 / n) * np.dot(X.T, errors)

                    # Gradient for bias (B)
                    gradient_B = (2 / n) * np.sum(errors)

                    return gradient_A, gradient_B
                else:                    
                    raise NotImplementedError("Activation function must be provided for the Linear classifier model.")

            case _: 
                raise NotImplementedError("This method should receive appropriate model name .")


        