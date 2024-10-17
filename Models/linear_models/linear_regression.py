import numpy as np
from Models.model import Model

class LinearRegression(Model):
    def __init__(self, cost_function):
        super().__init__()
        self.cost_function = cost_function
        self.params = None  # Parameters for weights (A)
        self.bias = None    # Parameter for bias (B)

    def predict(self, X):
        """Predict the output using the linear model: y = Ax + B"""
        return np.dot(X, self.params) + self.bias  # Include bias in predictions

    def compute_gradients(self, X, y):
        """Compute gradients for the model parameters."""
        prediction = self.predict(X)
        gradient_A, gradient_B = self.cost_function.compute_gradients(prediction, y, X, "LR")
        return gradient_A, gradient_B  # Return both gradients

    def fit(self, X, y, optimizer, num_epochs):
        """Train the model using the provided optimizer."""
        num_features = X.shape[1]
        self.params = np.zeros(num_features)  # Initialize weights (A) as a NumPy array
        self.bias = 0.0  # Initialize bias (B)

        for epoch in range(num_epochs):
            # Compute gradients
            gradient_A, gradient_B = self.compute_gradients(X, y)

            # Update parameters using optimizer
            self.params = optimizer.update(self.params, gradient_A)  # Update weights
            self.bias = optimizer.update(self.bias, gradient_B)

