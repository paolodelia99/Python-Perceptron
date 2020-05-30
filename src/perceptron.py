import random

import numpy as np


class Perceptron(object):

    def __init__(self, no_input: int, act_fn):
        """
        Perceptron contructor

        :argument no_input the numeber of the input of the percepton
        :argument act_fn the activation function of the percepton
        """
        self.no_input = no_input
        self.bias = random.random()
        self.weights = [np.random.rand(x) for x in range(no_input)]
        self.act_fn = act_fn

    def feedforward(self, a):
        weighted_sum = 0
        for w, x in zip(self.weights, a):
            weighted_sum = w * x

        return weighted_sum + self.bias

    def evaluate(self, inputs):
        return self.act_fn(self.feedforward(inputs))
