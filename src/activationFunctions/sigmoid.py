import numpy as np
from .activation_function import ActivationFunction


class Sigmoid(ActivationFunction):
    """Returns the sigmoid of x element-wise.
    """

    def __init__(self):
        """Construct of the sigmoid"""
        self.is_diff = True

    def compute(self, a):
        """Compute the output of the sigmoid operation

        :return the computed value given of the constructor
        """
        return 1 / (1 + np.exp(-a))

    def compute_derivative(self, a):
        """


        :return:
        """
        return self.compute(a) * (1 - self.compute(a))