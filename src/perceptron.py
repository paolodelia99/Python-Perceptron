import random
from src.activationFunctions.activation_function import ActivationFunction


class Perceptron(object):
    """Class representing the Percepton"""

    def __init__(self, no_input: int, act_fn: ActivationFunction, learining_rate: float):
        """
        Perceptron constructor

        :argument no_input the number of the input of the percepton
        :argument act_fn the activation function of the percepton
        """
        self.no_input = no_input
        self.bias = random.random()
        self.weights = [random.random() for _ in range(no_input)]
        self.act_fn = act_fn
        self.learning_rate = learining_rate

    def evaluate(self, inputs):
        weighted_sum = 0

        for w, x in zip(self.weights, inputs):
            weighted_sum = w * x

        weighted_sum += self.bias

        return self.act_fn.compute(weighted_sum)

    def update_weights(self, x, error):
        self.weights = [i - self.learning_rate * error * j for i, j in zip(self.w, x)]

    def train(self, data):

        pass