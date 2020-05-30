import nose
from src.perceptron import Perceptron
from src.activationFunctions.sgn import Sign

def test_precepton_101():
    """Test percepton istantiation"""
    p = Perceptron(3, Sign(), 0.1)
    nose.tools.assert_is_instance(p, Perceptron)

