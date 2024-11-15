
import numpy as np
from Models.utils.split_dataaset import split_dataset
from Models.utils.tree_data_structure import TreeNode
from CostFunctions.mse import MeanSquaredError  # Assuming MSE is implemented here

class DTRegression:
    def __init__(self, max_depth=None, min_samples_split=2, cost_function=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        # Use MSE as default if no cost function is provided
        self.cost_function = cost_function if cost_function else MeanSquaredError()
        self.root = None
    def _compute_cost(self, y_left, y_right):
        """
        Compute the total cost for a split using the provided cost function.
        """
        total_samples = len(y_left) + len(y_right)
        weighted_cost = (len(y_left) / total_samples) * self.cost_function.compute(np.mean(y_left), y_left) + \
                        (len(y_right) / total_samples) * self.cost_function.compute(np.mean(y_right), y_right)
        return weighted_cost
    def _best_split(self, X, y):
        best_feature = None
        best_threshold = None
        best_score = float('inf') 

        n_samples, n_features = X.shape
        if n_samples < self.min_samples_split:
            return None, None

        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                (X_left, y_left), (X_right, y_right) = split_dataset(X, y, feature_index, threshold)
                if len(y_left) > 0 and len(y_right) > 0:
                    # Calculate the score (e.g., MSE) for this split
                    score = self._compute_cost(y_left, y_right)
                    if score < best_score:
                        best_score = score
                        best_feature = feature_index
                        best_threshold = threshold

        return best_feature, best_threshold
    def _calculate_leaf_value(self, y):
        """
        Calculate the predicted value for a leaf (mean of y).
        """
        return np.mean(y)
    
    def fit(self, X, y, depth=0):
        best_feature, best_threshold = self._best_split(X, y)
        if best_feature is None or (self.max_depth is not None and depth >= self.max_depth):
            leaf_value = self._calculate_leaf_value(y)
            return TreeNode(value=leaf_value)

        (X_left, y_left), (X_right, y_right) = split_dataset(X, y, best_feature, best_threshold)
        left_subtree = self.fit(X_left, y_left, depth + 1)
        right_subtree = self.fit(X_right, y_right, depth + 1)

        return TreeNode(feature_index=best_feature, threshold=best_threshold,
                        left=left_subtree, right=right_subtree)
    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

    def predict(self, X):
        predictions = [self._traverse_tree(x, self.root) for x in X]
        return np.array(predictions)

    