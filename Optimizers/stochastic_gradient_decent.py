import numpy as np

from .optimizer import Optimizer

class SGD(Optimizer):
    def __init__(self, learning_rate=0.01,batch_size=1):
        super().__init__(learning_rate,batch_size)
    def update(self, params, gradients):
        params -= self.learning_rate * gradients
        return params

