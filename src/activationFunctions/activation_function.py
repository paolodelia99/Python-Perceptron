from abc import ABC, abstractmethod


class ActivationFunction(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def compute(self, a):
        pass

    @abstractmethod
    def compute_derivative(self, a):
        pass
