from abc import ABC, abstractmethod
import numpy as np

class AbstractKNN(ABC):
    def __init__(self, k=3, distance=None):
        self.k = k
        self.distance = distance

    def fit(self, X_train, y_train):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)

    def predict(self, X_test):
        predictions = [self.predict_point(x) for x in X_test]
        return np.array(predictions)

    @abstractmethod
    def predict_point(self, x):
        """Predict the label for a single point `x`.
           To be implemented in subclass."""
        pass
