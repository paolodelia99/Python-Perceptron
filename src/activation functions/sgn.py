from .activation_function import ActivationFunction

class sign(ActivationFunction):

    def __init__(self, a):

        super().__init__(a)

    def compute(self):
        if self.a < 0:
            return -1
        elif self.a == 0:
            return 0
        else:
            return 1