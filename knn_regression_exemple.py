import numpy as np
from Distances.cosine import CosineDistance
from Models.none_linear_models.knn_classification import KNNClassifier
from Models.none_linear_models.knn_regressor import KNNRegressor

# Example data
X_train = [[1, 2], [2, 3], [3, 4], [5, 6]]
y_train_regression = [1.5, 2.5, 3.5, 5.0]  # Continuous values for regression

# Distance object
cosine_distance = CosineDistance()
X_test = [[1, 3]]

# Regression
knn_regressor = KNNRegressor(k=3, distance=cosine_distance)
knn_regressor.fit(X_train, y_train_regression)
regression_predictions = knn_regressor.predict(X_test)

print("Regression predictions:", regression_predictions)