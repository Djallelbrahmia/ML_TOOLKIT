import numpy as np
from Models.linear_models import LinearClassifier
from CostFunctions import MeanSquaredError

# Create a larger synthetic dataset
# Generate 1000 samples with 2 features
np.random.seed(0)  # Set seed for reproducibility
X = np.random.randn(1000, 2)  # 1000 samples, 2 features

# Generate binary labels based on a linear combination of features with some noise
true_weights = np.array([1.5, -2.0])  # True weights for our linear function
bias = 0.5  # Bias term

# Calculate linear outputs and apply threshold for binary classification
linear_output = np.dot(X, true_weights) + bias
y = (linear_output > 0).astype(int)  # Binary labels: 0 or 1

# Initialize model and optimizer
cost_function = MeanSquaredError()
model = LinearClassifier(cost_function=cost_function, activation_function="sigmoid")

# Set the seed for reproducibility
seed = 42

# Fit the model
model.fit(X, y, num_epochs=10, batch_size=32, learning_rate=0.01, seed=seed, optimizer="sgd")

# Make predictions on the training data
predictions = model.predict(X)

# Print some predictions
print("Predictions on the training set:")
print(predictions[:10])  # Print the first 10 predictions