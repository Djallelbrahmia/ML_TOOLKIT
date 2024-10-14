from .optimizer import Optimizer
import numpy as np

class VGD(Optimizer):
    def __init__(self, learning_rate=0.01):
        super().__init__(learning_rate)
    def update(self, params, gradients):
        params -= self.learning_rate * gradients
        return params
    def train(self, model, data, labels, num_epochs, batch_size=1):
        num_samples = len(data)
        for epoch in range(num_epochs):
            indices = np.random.permutation(num_samples)
            data = data[indices]
            labels = labels[indices]
            gradients = model.compute_gradients(data, labels)  # Compute the gradients
            model.params = self.update(model.params, gradients)  # Update model parameters



    
