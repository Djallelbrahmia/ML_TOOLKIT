from .optimizer import Optimizer
import numpy as np


class RMSprop(Optimizer):
    def __init__(self, learning_rate=0.01, decay_rate=0.9, epsilon=1e-8, batch_size=1):
        super().__init__(learning_rate,batch_size)
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.average_squared_gradients = None  # Initialize average squared gradients

    def update(self, params, gradients):
      
        if self.average_squared_gradients is None:
            self.average_squared_gradients = np.zeros_like(gradients)

        # Update the moving average of the squared gradients
        self.average_squared_gradients = (self.decay_rate * self.average_squared_gradients) + ((1 - self.decay_rate) * (gradients ** 2))

        # Update parameters
        params -= self.learning_rate * gradients / (np.sqrt(self.average_squared_gradients) + self.epsilon)
        return params
    def train(self, model, data, labels, num_epochs):
       
        num_samples = len(data)

        for epoch in range(num_epochs):
            indices = np.random.permutation(num_samples)
            data = data[indices]
            labels = labels[indices]

            for i in range(0, num_samples, self.batch_size):
                batch_data = data[i:i + self.batch_size]
                batch_labels = labels[i:i + self.batch_size]
                gradients = model.compute_gradients(batch_data, batch_labels)
                model.params = self.update(model.params, gradients)


