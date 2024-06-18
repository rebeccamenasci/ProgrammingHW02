import numpy as np

def grad_log_barrier(x, t, grad_f, ineq_constraints, grad_ineq_constraints):
    grad_barrier = sum(grad_g(x) / -g(x) for g, grad_g in zip(ineq_constraints, grad_ineq_constraints))
    return t * grad_f(x) + grad_barrier

def hessian_log_barrier(x, t, hess_f, ineq_constraints, grad_ineq_constraints):
    hess_barrier = sum(np.outer(grad_g(x), grad_g(x)) / g(x)**2 for g, grad_g in zip(ineq_constraints, grad_ineq_constraints))
    return t * hess_f(x) + hess_barrier

def interior_pt(func, grad_f, hess_f, ineq_constraints, grad_ineq_constraints, eq_constraints_mat, eq_constraints_rhs, x0, t, mu, tol, max_iter):
    m, n = eq_constraints_mat.shape if eq_constraints_mat.size else (0, x0.size)
    x = np.copy(x0)
    path = [x]
    objective_values = [func(x)]
    outer_objective_values = []

    grad_f_barrier = lambda x: grad_log_barrier(x, t, grad_f, ineq_constraints, grad_ineq_constraints)
    hess_f_barrier = lambda x: hessian_log_barrier(x, t, hess_f, ineq_constraints, grad_ineq_constraints)

    for _ in range(max_iter):
        grad = grad_f_barrier(x)
        hess = hess_f_barrier(x)

        lhs = np.block([
            [hess, eq_constraints_mat.T if m > 0 else np.zeros((n, 0))],
            [eq_constraints_mat if m > 0 else np.zeros((0, n)), np.zeros((m, m))]
        ])
        rhs = -np.concatenate([grad, eq_constraints_mat @ x - eq_constraints_rhs])
        delta = np.linalg.solve(lhs, rhs)
        delta_x = delta[:n]

        alpha = 1
        while any(g(x + alpha * delta_x) <= 0 for g in ineq_constraints):
            alpha *= 0.9

        x += alpha * delta_x
        path.append(np.copy(x))
        objective_values.append(func(x))
        outer_objective_values.append(func(x))

        if np.linalg.norm(delta_x) < tol:
            return x, func(x), True, path, objective_values, outer_objective_values

        t *= mu

    return x, func(x), False, path, objective_values, outer_objective_values
