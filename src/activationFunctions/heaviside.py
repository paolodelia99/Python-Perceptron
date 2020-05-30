from .activation_function import ActivationFunction


class Heaviside(ActivationFunction):

    def __init__(self):
        self.is_diff = False

    def compute(self, a):
        return 1 if a >= 0 else 0
