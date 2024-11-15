import numpy as np
from Models.none_linear_models.decision_tree_classification import DTClassification
# Sample dataset for testing
X_train = np.array([[2, 3],
                    [1, 5],
                    [3, 6],
                    [6, 1],
                    [7, 2],
                    [8, 3]])
y_train = np.array([0, 1, 0, 1, 1, 0])

# Instantiate the Decision Tree Classifier
dt_classifier = DTClassification(max_depth=3, min_samples_split=2, criterion='gini')

# Fit the model to the training data
dt_classifier.root = dt_classifier.fit(X_train, y_train)

# Make predictions on the training set
predictions = dt_classifier.predict(X_train)

# Print the predictions and compare with actual labels
print("Predictions:", predictions)
print("Actual labels:", y_train)

# Simple accuracy check
accuracy = np.mean(predictions == y_train)
print("Accuracy:", accuracy)
