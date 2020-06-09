from src.functions.function import Function
from typing import Tuple


class QuadraticLoss(Function):

    def __init__(self):
        super(QuadraticLoss, self).__init__()
        self.is_diff = True

    def compute(self, y: Tuple[float, float]) -> float:
        return ((y[0] - y[1]) ** 2) / 2

    def compute_derivative(self, y: Tuple[float, float]):
        return y[0] - y[1]
