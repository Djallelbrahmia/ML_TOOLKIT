import numpy as np
from Models.model import Model
from Optimizers import OptimizerFactory

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

    def fit(self, X, y, num_epochs, batch_size,seed=None, optimizer="sgd", **kwargs):
        optimizer_factory = OptimizerFactory()
        slope_optimizer = optimizer_factory.create_optimizer(optimizer, **kwargs)
        bias_optimizer = optimizer_factory.create_optimizer(optimizer, **kwargs)

        num_samples, num_features = X.shape
        # Initialize weights and bias with a random seed if provided
        if seed is not None:
            np.random.seed(seed)
        self.params = np.random.randn(num_features) * 0.01  # Small random values for weights
        self.bias = np.random.randn() * 0.01  # Random bias initialization


        # Training loop over epochs
        for epoch in range(num_epochs):
            # Shuffle the dataset at the beginning of each epoch
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            # Mini-batch training loop
            for i in range(0, num_samples, batch_size):
                # Extract mini-batch
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                # Compute gradients for the mini-batch
                gradient_A, gradient_B = self.compute_gradients(X_batch, y_batch)

                # Update parameters using optimizer
                self.params = slope_optimizer.update(self.params, gradient_A)  # Update weights
                self.bias = bias_optimizer.update(self.bias, gradient_B)

            # Optionally, print loss or any diagnostic information
            print(f"Epoch {epoch + 1}/{num_epochs} completed.")
