import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../tests')))

from src.unconstrained_min import minimize
from examples import quadratic_example_1, quadratic_example_2, quadratic_example_3, rosenbrock, linear_function, boyds_function
from src.utils_opt import plot_objective_contours, plot_function_values

class TestUnconstrainedMinimization(unittest.TestCase):
    def setUp(self):
        self.obj_tol = 1e-12
        self.param_tol = 1e-8
        self.max_iter = 1000
        self.x0 = [1, 1]

    def run_optimization_tests(self, f, grad, hess):
        result_gd = minimize(f, self.x0, self.obj_tol, self.param_tol, self.max_iter, method='gradient_descent', grad=grad)
        result_nt = minimize(f, self.x0, self.obj_tol, self.param_tol, self.max_iter, method='newton', grad=grad, hess=hess) if hess else None

        self.assertTrue(result_gd[2], "Gradient Descent failed")
        if result_nt:
            self.assertTrue(result_nt[2], "Newton's Method failed")

        limits = [-2, 2, -2, 2]
        paths = [(result_gd[3], "Gradient Descent")]
        if result_nt:
            paths.append((result_nt[3], "Newton's Method"))

        plot_objective_contours(f, limits[:2], limits[2:], paths=paths)
        plot_function_values(
            (result_gd[3], "Gradient Descent"),
            (result_nt[3], "Newton's Method") if result_nt else None
        )

    def test_quadratic_example_1(self):
        f = lambda x: quadratic_example_1(x)[0]
        grad = lambda x: quadratic_example_1(x)[1]
        hess = lambda x: quadratic_example_1(x)[2]
        self.run_optimization_tests(f, grad, hess)

    def test_quadratic_example_2(self):
        f = lambda x: quadratic_example_2(x)[0]
        grad = lambda x: quadratic_example_2(x)[1]
        hess = lambda x: quadratic_example_2(x)[2]
        self.run_optimization_tests(f, grad, hess)

    def test_quadratic_example_3(self):
        f = lambda x: quadratic_example_3(x)[0]
        grad = lambda x: quadratic_example_3(x)[1]
        hess = lambda x: quadratic_example_3(x)[2]
        self.run_optimization_tests(f, grad, hess)

    def test_rosenbrock_function(self):
        f = lambda x: rosenbrock(x)[0]
        grad = lambda x: rosenbrock(x)[1]
        hess = lambda x: rosenbrock(x)[2]
        self.run_optimization_tests(f, grad, hess)

    def test_linear_function(self):
        f = lambda x: linear_function(x)[0]
        grad = lambda x: linear_function(x)[1]
        hess = lambda x: linear_function(x)[2]
        self.run_optimization_tests(f, grad, hess)

    def test_boyds_function(self):
        f = lambda x: boyds_function(x)[0]
        grad = lambda x: boyds_function(x)[1]
        hess = lambda x: boyds_function(x)[2]
        self.run_optimization_tests(f, grad, hess)

if __name__ == '__main__':
    unittest.main()
