from src.functions.function import Function


class MeanAbsErr(Function):

    def __init__(self):
        super(MeanAbsErr, self).__init__()
        self.is_diff = True

    def compute(self, y: tuple):
        return abs(y[0] - y[1])

    def compute_derivative(self, a) -> int:
        return -1
