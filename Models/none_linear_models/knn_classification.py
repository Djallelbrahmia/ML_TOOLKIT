import numpy as np
from Models.knnAbstract import AbstractKNN

class KNNClassifier(AbstractKNN):
    def predict_point(self, x):
        # Compute distances to all points in the training set
        distances = [self.distance.calculate(x, x_train) for x_train in self.X_train]
        
        # Get indices of the k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        
        # Get labels of the k nearest neighbors
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # Return the most common label (majority vote)
        return max(set(k_nearest_labels), key=k_nearest_labels.count)
