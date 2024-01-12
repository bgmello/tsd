import numpy as np
from scipy.optimize import minimize_scalar


def h(quantities, i, gamma):
    return (1 / (2 * quantities.N)) * np.linalg.norm(
        quantities.y
        - quantities.t
        - quantities.lambdas[i]
        * (np.exp(gamma) - 1)
        * (quantities.W[:, i] * quantities.W[:, i]),
        ord=2,
    ) ** 2


def loop(quantities, i):
    objective = lambda gamma: h(quantities, i, gamma)

    result = minimize_scalar(objective)

    optimal_gamma = result.x

    quantities.t = quantities.t + quantities.lambdas[i] * (
        np.exp(optimal_gamma) - 1
    ) * (quantities.W[:, i] * quantities.W[:, i])
    quantities.lambdas[i] = quantities.lambdas[i] * np.exp(optimal_gamma)

    return h(quantities, i, 0)
