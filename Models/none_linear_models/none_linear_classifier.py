import numpy as np
from Models.model import Model
from Optimizers import OptimizerFactory

class NoneLinearClassifier(Model) :
    def __init__(self,cost_function,activation_function,complexity):
        super().__init__()
        self.cost_function = cost_function
        self.complexity = complexity  # Degree of the polynomial
        self.params = None  # Parameters for weights
        self.bias = None    # Parameter for bias
        self.activation_function = activation_function
    def _polynomial_features(self, X):
            """Generate polynomial features from the input data."""
            return np.hstack([X ** i for i in range(1, self.complexity + 1)])
    def activation(self, z):
        """ Apply activation function to  output. """
        if self.activation_function == "th":
            return z > 0  # Threshold activation (binary classification)
        elif self.activation_function == "sigmoid":
            return 1 / (1 + np.exp(-z))  # Sigmoid activation
        else:
            raise ValueError("Unsupported activation function. Choose 'th' or 'sigmoid'.")
    def predict(self, X):
         
         """Predict the output using the non-linear model: y = A * X^complexity + B"""
         poly_features = self._polynomial_features(X)  # Generate polynomial features
         z = np.dot(poly_features, self.params) + self.bias
         activated_output = self.activation(z)
         if self.activation_function == "th":
            return activated_output.astype(int)  # For threshold, return 0 or 1
         elif self.activation_function == "sigmoid":
            return (activated_output > 0.5).astype(int)  # For sigmoid, return binary classification
    def compute_gradients(self, X, y):
             
             """Compute gradients for the model parameters."""
             poly_features = self._polynomial_features(X)  # Generate polynomial features
             z = np.dot(poly_features, self.params) + self.bias
             activated_output = self.activation(z)
             gradient_A, gradient_B = self.cost_function.compute_gradients(activated_output, y, poly_features,"NLC", self.activation_function)
             return gradient_A, gradient_B  # Return both gradients

    def fit(self, X, y, num_epochs, batch_size, seed=None, optimizer="sgd", **kwargs):

            optimizer_factory = OptimizerFactory()
            a_optimizer = optimizer_factory.create_optimizer(optimizer, **kwargs)
            bias_optimizer = optimizer_factory.create_optimizer(optimizer, **kwargs)

            num_samples, num_features = X.shape
                # Initialize polynomial features
            poly_features = self._polynomial_features(X)
            num_params = poly_features.shape[1]  # Number of polynomial terms

                # Initialize weights and bias with a random seed if provided
            if seed is not None:
                np.random.seed(seed)
            self.params = np.random.randn(num_params) * 0.01  # Small random values for weights
            self.bias = np.random.randn() * 0.01  # Random bias initialization
            if self.activation_function == "th":
                learning_rate = kwargs.get('learning_rate', 0.01)  # Default learning rate is 0.01
                #Perceptron training loop
                for epoch in range(num_epochs):
                    correct_predictions=0
                    for i in range(num_samples):

                        prediction= self.predict(X[i])
                        if(prediction)!=y[i]:
                            self.params+=learning_rate*(y[i]-prediction)*poly_features[i]
                            self.bias += learning_rate * (y[i] - prediction)
                        if prediction == y[i]:
                            correct_predictions += 1
                    if correct_predictions == num_samples:
                        print(f"Early stopping at epoch {epoch + 1} with all samples classified correctly.")
                        break  # Stop early if all samples are classified correctly
            else  :           
            # Gradient descent training loop  activation

                  
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
                        self.params = a_optimizer.update(self.params, gradient_A)  # Update weights
                        self.bias = bias_optimizer.update(self.bias, gradient_B)

                        # Optionally, print loss or any diagnostic information
                    print(f"Epoch {epoch + 1}/{num_epochs} completed.")

