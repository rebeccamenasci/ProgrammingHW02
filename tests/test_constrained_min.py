import sys
import os
import unittest
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../tests')))

from src.constrained_min import interior_pt
from examples import qp_example, lp_example

def plot_3d_feasible_region(func, ineq_constraints, path, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X = np.linspace(0, 1, 30)
    Y = np.linspace(0, 1, 30)
    X, Y = np.meshgrid(X, Y)
    Z = 1 - X - Y
    Z[Z < 0] = np.nan 
    ax.plot_surface(X, Y, Z, alpha=0.5, rstride=100, cstride=100)
    path = np.array(path)
    ax.plot(path[:, 0], path[:, 1], path[:, 2], marker='o', color='blue', label='Path')
    ax.scatter(path[-1, 0], path[-1, 1], path[-1, 2], c='yellow', label='Final candidate')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_zlim([0, 1])
    plt.title(title)
    plt.legend()
    plt.show()



def plot_2d_feasible_region(func, ineq_constraints, path, title):
    fig, ax = plt.subplots()
    x = np.linspace(0, 3, 400)
    y = np.linspace(0, 2, 400)
    X, Y = np.meshgrid(x, y)
    for g in ineq_constraints:
        Z = np.vectorize(lambda a, b: g(np.array([a, b])))(X, Y)
        ax.contour(X, Y, Z, levels=[0], colors='r')
    ax.plot([p[0] for p in path], [p[1] for p in path], marker='o')
    ax.scatter(path[-1][0], path[-1][1], c='r')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.title(title)
    plt.show()

class TestConstrainedMinimization(unittest.TestCase):
    def setUp(self):
        self.t = 1
        self.mu = 10
        self.param_tol = 1e-8
        self.max_iter = 100

    def run_optimization_tests(self, func, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, x0, plot_title, test_name):
        grad_f = lambda x: np.array([2 * x[0], 2 * x[1], 2 * (x[2] + 1)]) if len(x) == 3 else np.array([-1, -1])
        hess_f = lambda x: np.diag([2, 2, 2]) if len(x) == 3 else np.zeros((2, 2))

        grad_ineq_constraints = [
            lambda x: np.array([1, 0, 0]) if len(x) == 3 else np.array([1, -1]),  # Gradient of x >= 0 or y >= -x + 1
            lambda x: np.array([0, 1, 0]) if len(x) == 3 else np.array([0, -1]),  # Gradient of y >= 0 or y <= 1
            lambda x: np.array([0, 0, 1]) if len(x) == 3 else np.array([-1, 0]),  # Gradient of z >= 0 or x <= 2
            lambda x: np.array([0, 0, 1]) if len(x) == 3 else np.array([0, 1])    # Gradient of z <= 1 (if needed) or y >= 0
        ]

        x, fx, success, path, objective_values, outer_objective_values = interior_pt(
            func, grad_f, hess_f,
            ineq_constraints, grad_ineq_constraints,
            eq_constraints_mat, eq_constraints_rhs, x0, self.t, self.mu, self.param_tol, self.max_iter
        )
        self.assertTrue(success, f"{test_name} failed to converge")

        if plot_title:
            if '3D' in plot_title:
                plot_3d_feasible_region(func, ineq_constraints, path, plot_title)
            else:
                plot_2d_feasible_region(func, ineq_constraints, path, plot_title)

        plt.plot(objective_values, label="Objective Value")
        plt.plot(outer_objective_values, label="Outer Objective Value")
        plt.xlabel("Iteration")
        plt.ylabel("Objective Value")
        plt.title(f"Objective Value vs Iteration ({test_name})")
        plt.legend()
        plt.show()

    def test_qp(self):
        func, ineq_constraints_qp, eq_constraints_mat_qp, eq_constraints_rhs_qp, x0 = qp_example()
        self.run_optimization_tests(func, ineq_constraints_qp, eq_constraints_mat_qp, eq_constraints_rhs_qp, x0, "3D Path of QP Example", "QP Example")

    def test_lp(self):
        func, ineq_constraints_lp, eq_constraints_mat_lp, eq_constraints_rhs_lp, x0 = lp_example()
        self.run_optimization_tests(func, ineq_constraints_lp, eq_constraints_mat_lp, eq_constraints_rhs_lp, x0, "2D Path of LP Example", "LP Example")

if __name__ == '__main__':
    unittest.main()
