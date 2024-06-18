import numpy as np

def line_search(f, x, direction, alpha=1.0, beta=0.8, sigma=1e-4):
    while f(x + alpha * direction) > f(x) + sigma * alpha * np.dot(direction, direction):
        alpha *= beta
    return alpha

def optimize_newton_method(f, grad_f, hess_f, x0, obj_tol, param_tol, max_iter):
    x = np.array(x0, dtype=float)
    path = [x.copy()]
    for _ in range(max_iter):
        fx, grad, hess = f(x), grad_f(x), hess_f(x)
        if np.linalg.norm(grad) < obj_tol:
            return x, fx, True, path
        if np.linalg.cond(hess) > 1 / np.finfo(float).eps:
            print("Hessian is near singular.")
            return x, fx, False, path
        direction = -np.linalg.solve(hess, grad)
        alpha = line_search(lambda x: f(x), x, direction)
        x += alpha * direction
        path.append(x.copy())
        if np.linalg.norm(alpha * direction) < param_tol:
            return x, fx, True, path
    return x, fx, False, path

def optimize_gradient_descent(f, grad_f, x0, obj_tol, param_tol, max_iter):
    x = np.array(x0, dtype=float)
    path = [x.copy()]
    for _ in range(max_iter):
        grad = grad_f(x)
        if np.linalg.norm(grad) < obj_tol:
            return x, f(x), True, path
        direction = -grad
        alpha = line_search(f, x, direction)
        x += alpha * direction
        path.append(x.copy())
        if np.linalg.norm(alpha * direction) < param_tol:
            return x, f(x), True, path
    return x, f(x), False, path

def minimize(f, x0, obj_tol, param_tol, max_iter, method='gradient_descent', grad=None, hess=None):
    if method == 'gradient_descent':
        return optimize_gradient_descent(f, grad, x0, obj_tol, param_tol, max_iter)
    elif method == 'newton':
        return optimize_newton_method(f, grad, hess, x0, obj_tol, param_tol, max_iter)
    else:
        raise ValueError(f"Unknown method {method}")
