import nose
from src.activationFunctions.sgn import Sign
from src.activationFunctions.heaviside import Heaviside
from src.activationFunctions.indentity import Identity
from src.activationFunctions.sigmoid import Sigmoid
from src.activationFunctions.activation_function import ActivationFunction


def test_act_fn_101():
    """Test percepton istantiation"""
    fn = ActivationFunction()
    nose.tools.assert_is_instance(fn, ActivationFunction)
    nose.tools.assert_equal(fn.is_diff, None)


def test_sing_fn():
    """Test sign function"""
    fn = Sign()
    nose.tools.assert_is_instance(fn, ActivationFunction)
    nose.tools.assert_is_instance(fn, Sign)
    nose.tools.assert_false(fn.is_diff)
    nose.tools.assert_equal(fn.compute(10), 1)
    nose.tools.assert_equal(fn.compute(0), 0)
    nose.tools.assert_equal(fn.compute(-10), -1)


def test_heavside_fn():
    """Test Heaviside function"""
    fn = Heaviside()
    nose.tools.assert_is_instance(fn, ActivationFunction)
    nose.tools.assert_is_instance(fn, Heaviside)
    nose.tools.assert_false(fn.is_diff)
    nose.tools.assert_equal(fn.compute(10), 1)
    nose.tools.assert_equal(fn.compute(0), 1)
    nose.tools.assert_equal(fn.compute(-10), 0)


def test_identity_fn():
    """Test Identity function"""
    fn = Identity()
    nose.tools.assert_is_instance(fn, ActivationFunction)
    nose.tools.assert_is_instance(fn, Identity)
    nose.tools.assert_true(fn.is_diff)
    nose.tools.assert_equal(fn.compute(10), 10)
    nose.tools.assert_equal(fn.compute(0), 0)
    nose.tools.assert_equal(fn.compute(-10), -10)
    nose.tools.assert_equal(fn.compute_derivative(), 1)


def test_sigmoid_fn():
    """Test the sigmoid function"""
    fn = Sigmoid()
    nose.tools.assert_is_instance(fn, ActivationFunction)
    nose.tools.assert_is_instance(fn, Sigmoid)
    nose.tools.assert_equal(fn.compute(0), 0.5)
    nose.tools.assert_equal(fn.compute_derivative(0), 0.25)
