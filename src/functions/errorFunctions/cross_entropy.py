from src.functions.function import Function
import numpy as np
import math
from typing import Tuple


class CrossEntropy(Function):
    """
    Class representing the cross entropy cost function
    """

    def __init__(self):
        super(CrossEntropy, self).__init__()
        self.is_diff = True

    def compute(self, y: Tuple[float, float]) -> float:
        res = 0 - (y[0] * math.log1p(y[1]) + (1 - y[0]) * math.log1p(1 - y[1]))
        return res

    def compute_derivative(self, y: Tuple[float, float]) -> float:
        return (y[1] - y[0]) / ((1 - y[1]) * y[1])