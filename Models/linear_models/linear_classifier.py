import numpy as np
from Models.model import Model
from Optimizers import OptimizerFactory

class LinearClassifier(Model):
    def __init__(self, cost_function, activation_function):
        super().__init__()
        self.cost_function = cost_function
        self.params = None  # Parameters for weights (A)
        self.bias = None    # Parameter for bias (B)
        self.activation_function = activation_function

    def activation(self, z):
        """ Apply activation function to the linear output. """
        if self.activation_function == "th":
            return z > 0  # Threshold activation (binary classification)
        elif self.activation_function == "sigmoid":
            return 1 / (1 + np.exp(-z))  # Sigmoid activation
        else:
            raise ValueError("Unsupported activation function. Choose 'th' or 'sigmoid'.")

    def compute_gradients(self, X, y):
        """ Compute gradients for the model parameters. """
        z = np.dot(X, self.params) + self.bias
        activated_output = self.activation(z)

        gradient_A, gradient_B = self.cost_function.compute_gradients(activated_output, y, X, model="LC", 
                                                                      activation_function=self.activation_function)
        return gradient_A, gradient_B

    def predict(self, X):
        """ Predict the class labels based on the input features. """
        z = np.dot(X, self.params) + self.bias
        return self.activation(z)
        
    def fit(self, X, y, num_epochs, batch_size, seed=None, optimizer="sgd", **kwargs):
        optimizer_factory = OptimizerFactory()
        slope_optimizer = optimizer_factory.create_optimizer(optimizer, **kwargs)
        bias_optimizer = optimizer_factory.create_optimizer(optimizer, **kwargs)

        num_samples, num_features = X.shape
        # Initialize weights and bias with a random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize weights and bias randomly
        self.params = np.random.randn(num_features) * 0.01  # Small random values for weights
        self.bias = np.random.randn() * 0.01  # Random bias initialization

        if self.activation_function == "th":
            learning_rate = kwargs.get('learning_rate', 0.01)  # Default learning rate is 0.01

            # Perceptron training loop
            for epoch in range(num_epochs):
                correct_predictions = 0
                for i in range(num_samples):
                    # Calculate the linear output
                    z = np.dot(X[i], self.params) + self.bias
                    prediction = self.activation(z)  # Apply threshold activation

                    # Update weights if prediction is wrong
                    if prediction != y[i]:
                        # Perceptron update rule
                        self.params += learning_rate * (y[i] - prediction) * X[i]
                        self.bias += learning_rate * (y[i] - prediction)
                    if prediction == y[i]:
                        correct_predictions += 1

                if correct_predictions == num_samples:
                    print(f"Early stopping at epoch {epoch + 1} with all samples classified correctly.")
                    break  # Stop early if all samples are classified correctly
        else:
            # Gradient descent training loop for sigmoid activation
            for epoch in range(num_epochs):
                for i in range(0, num_samples, batch_size):
                    X_batch = X[i:i + batch_size]
                    y_batch = y[i:i + batch_size]

                    # Compute gradients for the mini-batch
                    gradient_A, gradient_B = self.compute_gradients(X_batch, y_batch)

                    # Update parameters using optimizer
                    self.params = slope_optimizer.update(self.params, gradient_A)  # Update weights
                    self.bias = bias_optimizer.update(self.bias, gradient_B)
