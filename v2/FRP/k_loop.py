import numpy as np
from scipy.optimize import minimize_scalar
from quantities import Quantities

def loop(quantities: Quantities, k):
    v_k = v(quantities, k)
    optimal_gamma = find_optimal_gamma(quantities, v_k, k)

    s_result = s(quantities, v_k, k, optimal_gamma)
    z_result = z(quantities, v_k, k, optimal_gamma)

    quantities.W[:, k] = quantities.X.T @ (s_result + quantities.UQ[:, k])  # X.T shape = N, n
    quantities.t = (
        quantities.t + 2 * quantities.lambdas[k] * quantities.W[:, k] * z_result - quantities.lambdas[k] * (z_result * z_result)
    )
    quantities.UQ[:, k] += s_result

    return h(quantities, v(quantities, k), k, 0)

def v(quantities, k):
    t_minus_y_W_k = (quantities.t - quantities.y) * quantities.W[:, k]
    v_hat = (2 * quantities.lambdas[k] / quantities.N) * (
        quantities.X @ (t_minus_y_W_k) - quantities.UQ @ (quantities.W.T @ (t_minus_y_W_k))
    )
    if np.linalg.norm(v_hat, ord=2) == 0:
        return np.zeros_like(v_hat)
    return v_hat / np.linalg.norm(v_hat, ord=2)

def find_optimal_gamma(quantities, v_k, k):
    objective = lambda gamma: h(quantities, v_k, k, gamma)
    result = minimize_scalar(objective, bounds=(0, 2 * np.pi), method="bounded")

    return result.x

def h(quantities, v_k, k, gamma):
    z_result = z(quantities, v_k, k, gamma)
    lambda_z = quantities.lambdas[k] * z_result
    res = quantities.y - quantities.t - 2 * quantities.W[:, k] * lambda_z - lambda_z * z_result
    return (1 / (2 * quantities.N)) * np.linalg.norm(res, ord=2) ** 2

def s(quantities, v_k, k, gamma):
    cos = np.cos(gamma)
    sin = np.sin(gamma)
    return (cos - 1) * quantities.UQ[:, k] + sin * v_k

def z(quantities, v_k, k, gamma):
    cos = np.cos(gamma)
    sin = np.sin(gamma)
    return (cos - 1) * quantities.W[:, k] + sin * quantities.X.T @ v_k