import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def plot_contours(f, x_range, y_range, paths=None, title="Contour Plot"):
    x = np.linspace(x_range[0], x_range[1], 400)
    y = np.linspace(y_range[0], y_range[1], 400)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[f(np.array([X[i, j], Y[i, j]]))[0] for j in range(len(x))] for i in range(len(y))])
    
    plt.figure(figsize=(8, 6))
    cp = plt.contour(X, Y, Z, levels=np.logspace(-1, 3, 35), norm=LogNorm(), cmap=plt.cm.jet)
    plt.colorbar(cp)
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')

    if paths:
        for path, label in paths:
            path = np.array(path)
            plt.plot(path[:, 0], path[:, 1], marker='o', label=label)
        plt.legend()

    plt.show()

# Example usage:
# Define a simple quadratic function for plotting
def example_func(x):
    return np.sum(x**2), 2*x

# Call the plotting function
plot_contours(example_func, [-5, 5], [-5, 5])


def plot_function_values(paths, title="Function Values Over Iterations"):
    plt.figure(figsize=(8, 6))
    for path, label in paths:
        values = [f[1] for f in path]
        plt.plot(values, label=label)
    
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Function Value')
    plt.legend()
    plt.yscale('log')  # Optional: set to log scale if values vary widely
    plt.show()

# Example usage:
# Suppose paths contains iteration paths of different optimization methods
# plot_function_values([([example_func(np.array([1, 2*i])) for i in range(10)], "Method A")])
