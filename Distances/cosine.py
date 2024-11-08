import numpy as np
from .distance import Distance

class CosineDistance(Distance):
    def calculate(self, point1, point2):
        dot_product = np.dot(point1, point2)
        norm_a = np.linalg.norm(point1)
        norm_b = np.linalg.norm(point2)
        return 1 - (dot_product / (norm_a * norm_b))