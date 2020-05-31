import nose
from src.perceptron import Perceptron
from src.activationFunctions.sgn import Sign
from src.activationFunctions.heaviside import Heaviside
from src.activationFunctions.activation_function import ActivationFunction


def test_precepton_101():
    """Test percepton istantiation"""
    p = Perceptron(3, Sign(), 0.1)
    nose.tools.assert_is_instance(p, Perceptron)
    nose.tools.assert_is_instance(p.learning_rate, float)
    nose.tools.assert_is_instance(p.act_fn, ActivationFunction)
    nose.tools.assert_is_instance(p.weights, list)
    nose.tools.assert_is_instance(p.bias, float)


def test_preceptron_102():
    dataset = [[2.7810836, 2.550537003, 0],
               [1.465489372, 2.362125076, 0],
               [3.396561688, 4.400293529, 0],
               [1.38807019, 1.850220317, 0],
               [3.06407232, 3.005305973, 0],
               [7.627531214, 2.759262235, 1],
               [5.332441248, 2.088626775, 1],
               [6.922596716, 1.77106367, 1],
               [8.675418651, -0.242068655, 1],
               [7.673756466, 3.508563011, 1]]
    p = Perceptron(2, Heaviside(), 0.1)
    p.train(dataset, 10)

    for d in dataset:
        nose.tools.assert_equal(p.evaluate([d[0],d[1]]), d[2])

