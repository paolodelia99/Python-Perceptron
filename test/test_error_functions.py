import nose
from src.functions.function import Function
from src.functions.lossFunctions.mean_abs_error import MeanAbsErr
from src.functions.lossFunctions.quadratic_loss import QuadraticLoss
from src.functions.lossFunctions.cross_entropy import CrossEntropy


def test_mean_square_error():
    fn = QuadraticLoss()
    nose.tools.assert_is_instance(fn, Function)
    nose.tools.assert_is_instance(fn, QuadraticLoss)
    nose.tools.assert_true(fn.is_diff)
    nose.tools.assert_equals(fn.compute((10, 5)), 12.5)
    nose.tools.assert_equals(fn.compute_derivative((10, 5)), 5)


def test_mean_abs_error():
    fn = MeanAbsErr()
    nose.tools.assert_is_instance(fn, Function)
    nose.tools.assert_is_instance(fn, MeanAbsErr)
    nose.tools.assert_true(fn.is_diff)
    nose.tools.assert_equals(fn.compute((10, 5)), 5)
    nose.tools.assert_equals(fn.compute_derivative((10, 5)), -1)


def test_cross_entropy():
    fn = CrossEntropy()
    nose.tools.assert_is_instance(fn, Function)
    nose.tools.assert_is_instance(fn, CrossEntropy)
    nose.tools.assert_true(fn.is_diff)
    nose.tools.assert_equals(fn.compute((0.75, 0.5)), -0.4054651081081644)
    nose.tools.assert_equals(fn.compute_derivative((0.75, 0.50)), -1.0)
