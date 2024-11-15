import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from Models.none_linear_models.decision_tree_classification import DTClassification
print("hey")
# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Instantiate the Decision Tree Classifier
dt_classifier = DTClassification(max_depth=3, min_samples_split=5, criterion='gini')

# Fit the model to the training data
dt_classifier.root = dt_classifier.fit(X_train, y_train)

# Make predictions on the test set
predictions = dt_classifier.predict(X_test)

# Print the predictions and compare with actual labels
print("Predictions:", predictions)
print("Actual labels:", y_test)

# Simple accuracy check
accuracy = np.mean(predictions == y_test)
print("Accuracy:", accuracy)
