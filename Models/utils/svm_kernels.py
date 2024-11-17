import numpy as np

def linear_kernel(x1, x2):
    """
    Linear kernel: k(x1, x2) = x1ᵀx2
    """
    return np.dot(x1, x2)

def rbf_kernel(gamma=1.0):
    """
    RBF (Gaussian) kernel: k(x1, x2) = exp(-gamma ||x1-x2||²)
    :param gamma: Kernel coefficient (default=1.0)
    :return: RBF kernel function
    """
    def kernel(x1, x2):
        return np.exp(-gamma * np.sum((x1 - x2) ** 2))
    return kernel

def polynomial_kernel(degree=2, gamma=1.0, coef0=0.0):
    """
    Polynomial kernel: k(x1, x2) = (gamma * x1ᵀx2 + coef0)^degree
    :param degree: Polynomial degree (default=2)
    :param gamma: Kernel coefficient (default=1.0)
    :param coef0: Independent term (default=0.0)
    :return: Polynomial kernel function
    """
    def kernel(x1, x2):
        return (gamma * np.dot(x1, x2) + coef0) ** degree
    return kernel