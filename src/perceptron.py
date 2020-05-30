import random
from src.activationFunctions.activation_function import ActivationFunction
import numpy as np


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
        """
        Return the prediction of the perceptron

        :param inputs: the inputs feature
        :return: the evaluation made by the percepton
        """
        weighted_sum = 0

        for w, x in zip(self.weights, inputs):
            weighted_sum = w * x

        weighted_sum += self.bias

        return self.act_fn.compute(weighted_sum)

    def update_weights(self, x, error):
        """
        Update the perceptron weight based on the inputs and the error

        :param x: The inputs
        :param error: the estimated error
        """
        # Check if the activation function is differentiable
        if self.act_fn.is_diff:
            self.weights = [
                i - self.learning_rate * error * self.act_fn.compute_derivative(np.dot(self.weights, x) + self.bias) * j
                for i, j in zip(self.w, x)]  # fixme da sistemare
        else:
            self.weights = [i - self.learning_rate * error * j for i, j in zip(self.w, x)]

    def train(self, data):
        """
        Just a draft f what would be the training algo

        :argument data the data used to train the precepton
        """
        for d, e in data:
            res = self.evaluate(d)
            if e != res:
                error = e - res
                self.update_weights(d, error)
