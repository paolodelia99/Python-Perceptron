from .activation_function import ActivationFunction


class Identity(ActivationFunction):

    def __init__(self):
        self.is_diff = False

    def compute(self, a):
        return self.a
