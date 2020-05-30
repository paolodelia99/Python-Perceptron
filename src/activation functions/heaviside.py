from .activation_function import ActivationFunction


class Heaviside(ActivationFunction):

    def __init__(self, a):
        super().__init__(a)

    def compute(self):
        return 1 if self.a >= 0 else 0
