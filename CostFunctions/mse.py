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

    def compute_gradients(self, predictions, targets, X, model, activation_function=None):
        """
        Compute the gradients of the Mean Squared Error.
        
        :param predictions: Predicted values from the model
        :param targets: True target values
        :param model: model to return gradient related to that model 
        :param X: Input data

        :return: Gradients of the MSE
        """
        n = len(targets)  # Number of samples
        errors = predictions - targets  # Error term

        match model:
            case "LR":
                # Gradient for weights A
                gradient_A = (2 / n) * np.dot(X.T, errors)  # Gradient w.r.t. weights A

                # Gradient for bias B
                gradient_B = (2 / n) * np.sum(errors)  # Gradient w.r.t. bias B

                return gradient_A, gradient_B
            
            case "NLR":
                # Gradient for weights A
                gradient_A = (2 / n) * np.dot(X.T, errors)  # Gradient w.r.t. weights A

                # Gradient for bias B
                gradient_B = (2 / n) * np.sum(errors)  # Gradient w.r.t. bias B

                return gradient_A, gradient_B
            
            case "LC":
                if activation_function == "sigmoid":
                    # Derivative of the sigmoid function
                    sigmoid_derivative = predictions * (1 - predictions)
                    adjusted_errors = errors * sigmoid_derivative  # Adjust errors by the derivative

                    # Gradient for weights (A)
                    gradient_A = (2 / n) * np.dot(X.T, adjusted_errors)

                    # Gradient for bias (B)
                    gradient_B = (2 / n) * np.sum(adjusted_errors)  # Use adjusted_errors here

                    return gradient_A, gradient_B
                else:
                    raise NotImplementedError("Activation function must be provided for the Linear classifier model.")
            
            case "NLC":
                if activation_function == "sigmoid":
                    # Derivative of the sigmoid function
                    sigmoid_derivative = predictions * (1 - predictions)
                    adjusted_errors = errors * sigmoid_derivative  # Adjust errors by the derivative
                    # Gradient for weights (since X already contains polynomial terms)
                    gradient_A = (2 / n) * np.dot(X.T, adjusted_errors)

                    # Gradient for the bias term
                    gradient_B = (2 / n) * np.sum(adjusted_errors)  # Use adjusted_errors here

                    return gradient_A, gradient_B
                else:
                    raise NotImplementedError("Activation function must be provided for the Non-Linear classifier model.")

            case _: 
                raise NotImplementedError("This method should receive appropriate model name.")
