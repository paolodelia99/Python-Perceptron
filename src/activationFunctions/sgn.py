from .activation_function import ActivationFunction


class Sign(ActivationFunction):

    def __init__(self):
        self.is_diff = False

    def compute(self, a):
        if a < 0:
            return -1
        elif a == 0:
            return 0
        else:
            return 1

    def compute_derivative(self):
        pass
