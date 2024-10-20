from .optimizer import Optimizer
import numpy as np

class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, batch_size=1):

        super().__init__(learning_rate,batch_size)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None  # First moment vector
        self.v = None  # Second moment vector
        self.t = 0     # Time step

    def update(self, params, gradients):

        # Initialize moment vectors if they are None
        if self.m is None:
            self.m = np.zeros_like(gradients)
        if self.v is None:
            self.v = np.zeros_like(gradients)

        # Increment time step
        self.t += 1

        # Update biased first moment estimate
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradients

        # Update biased second moment estimate
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradients ** 2)

        # Correct bias for first moment
        m_hat = self.m / (1 - self.beta1 ** self.t)

        # Correct bias for second moment
        v_hat = self.v / (1 - self.beta2 ** self.t)

        # Update parameters
        params -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return params


 
