from .optimizer import Optimizer
import numpy as np

class Momentum(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.9, batch_size=1):
        super().__init__(learning_rate,batch_size)
        self.momentum = momentum
        self.velocity = None  # Initialize velocity
        
    def update(self, params, gradients):
        if self.velocity is None:
            self.velocity = np.zeros_like(params)       
        self.velocity = self.momentum * self.velocity + (1 - self.momentum) * gradients
        params -= self.learning_rate * self.velocity
        return params
