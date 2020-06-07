from .activation_function import ActivationFunction


class ReLU(ActivationFunction):
    """Returns the sigmoid of x element-wise.
    """

    def __init__(self):
        """Construct of the sigmoid"""
        super().__init__()
        self.is_diff = True

    def compute(self, a):
        """
        Compute the sigmoid function on the given value

        :return the computed value given of the constructor
        """
        return a if a > 0 else 0

    def compute_derivative(self, a):
        """
        Compute the derivative of the sigmoid function on the given value

        :return: the value calculated on the derivative of the sigmoid
        """
        return 1 if a > 0 else 0
