from abc import ABC, abstractmethod


class ActivationFunction(ABC):

    def __init__(self, a):
        self.a = a

    @abstractmethod
    def compute(self):
        pass
