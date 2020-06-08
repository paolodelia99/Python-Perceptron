from src.functions.function import Function


class MeanSquareErr(Function):

    def __init__(self):
        super(MeanSquareErr, self).__init__()
        self.is_diff = True

    def compute(self, y: tuple) -> float:
        return ((y[0] - y[1]) ** 2) / 2

    def compute_derivative(self, y: tuple):
        return y[0] - y[1]
