import numpy as np
from Models.none_linear_models.decision_tree_regression import DTRegression
from CostFunctions import MeanSquaredError

reg_tree = DTRegression(max_depth=3, min_samples_split=2, cost_function=MeanSquaredError())
X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
y_train = np.array([1.1, 2.5, 3.7, 4.2, 5.8])  # Slightly different target values to prevent perfect fit


# Fit the regression tree
reg_tree.root = reg_tree.fit(X_train, y_train)

# Predict on the training data
predictions = reg_tree.predict(X_train)
print("Predictions:", predictions)
mse_calculator = MeanSquaredError()
mse = mse_calculator.compute(predictions, y_train)
print("Mean Squared Error:", mse)