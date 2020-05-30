import numpy as np
from .activation_function import ActivationFunction


class sigmoid(ActivationFunction):
    """Returns the sigmoid of x element-wise.
    """

    def __init__(self, a):
        """Construct sigmoid

        Args:
          a: Input node
        """
        super().__init__(a)

    def compute(self):
        """Compute the output of the sigmoid operation

        Args:
          a_value: Input value
        """
        return 1 / (1 + np.exp(-self.a))
