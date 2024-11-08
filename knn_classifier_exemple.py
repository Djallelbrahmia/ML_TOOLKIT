import numpy as np
from Distances.euclidian import EuclideanDistance
from Models.none_linear_models.knn_classification import KNNClassifier
from Models.none_linear_models.knn_regressor import KNNRegressor

# Example data
X_train = [[1, 2], [2, 3], [3, 4], [5, 6]]
y_train_classification = [0, 1, 1, 0]  # Categorical labels for classification

# Distance object
euclidean_distance = EuclideanDistance()

# Classification
knn_classifier = KNNClassifier(k=3, distance=euclidean_distance)
knn_classifier.fit(X_train, y_train_classification)
X_test = [[1, 3]]
classification_predictions = knn_classifier.predict(X_test)
print("Classification predictions:", classification_predictions)
