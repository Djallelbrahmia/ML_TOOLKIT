from .optimizer import Optimizer
import numpy as np

class VGD(Optimizer):
    def __init__(self, learning_rate=0.01):
        super().__init__(learning_rate)
    def update(self, params, gradients):
        params -= self.learning_rate * gradients
        return params
