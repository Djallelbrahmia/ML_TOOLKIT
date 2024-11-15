import numpy as np
from sklearn.datasets import fetch_california_housing
from Models.none_linear_models.decision_tree_regression import DTRegression
from CostFunctions import MeanSquaredError

# Initialize the decision tree regressor
reg_tree = DTRegression(max_depth=3, min_samples_split=2, cost_function=MeanSquaredError())

# Load the California housing dataset
california_housing = fetch_california_housing()
X_train = california_housing.data[:100, :]  # Using only the first 100 samples for training
y_train = california_housing.target[:100]

# Fit the regression tree
reg_tree.root = reg_tree.fit(X_train, y_train)

# Predict on the training data
predictions = reg_tree.predict(X_train)
print("Predictions:", predictions[:5])  # Print first 5 predictions to check

# Calculate Mean Squared Error
mse_calculator = MeanSquaredError()
mse = mse_calculator.compute(predictions, y_train)
print("Mean Squared Error:", mse)
