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
        :argument learining_rate: the learning rate of the perceptron
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
        weighted_sum = np.dot(self.weights, inputs)

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
                for i, j in zip(self.w, x)]
        else:
            self.weights = [i + self.learning_rate * error * j for i, j in zip(self.weights, x)]

    def train(self, data, n_epoch=10):
        """
        Just a draft f what would be the training algo

        :argument n_epoch: number of iteration
        :argument data the data used to train the precepton
        """
        for epoch in range(n_epoch):
            for row in data:
                prediction = self.evaluate(row[0: len(self.weights)])
                error = row[-1] - prediction
                self.bias = self.bias + self.learning_rate * error  # update the bias
                self.update_weights(row, error)  # update the weights
