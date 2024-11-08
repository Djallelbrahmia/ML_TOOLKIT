import numpy as np
from .distance import Distance

class EuclideanDistance(Distance):
    def calculate(self, point1, point2):
        return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))