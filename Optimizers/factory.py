from .adam import Adam
from .momentum_gradient_decent import Momentum
from .rmsprop import RMSprop
from .stochastic_gradient_decent import SGD
from .vanilla_gradient_decent import VGD

class OptimizerFactory:
    """
    Factory class to create instances of optimizers.
    """
    @staticmethod
    def create_optimizer(optimizer_name, **kwargs):
        """
        Factory method to create and return an optimizer instance.
        :param optimizer_name: Name of the optimizer ('adam', 'rmsprop', etc.)
        :param kwargs: Parameters for the optimizer
        :return: Instance of the requested optimizer
        """
        optimizers = {
            'adam': Adam,
            'rmsprop': RMSprop,
            'momentum':Momentum,
            "sgd":SGD,
            "vgd":VGD
            # You can add more optimizers here
        }

        optimizer_class = optimizers.get(optimizer_name.lower())

        if optimizer_class is None:
            raise ValueError(f"Optimizer '{optimizer_name}' is not recognized.")

        return optimizer_class(**kwargs)
