from .activation_function import ActivationFunction


class Identity(ActivationFunction):

    def __init__(self, a):
        super().__init__(a)

    def compute(self):
        return self.a
