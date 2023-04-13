import numpy as np
import random
from scipy.optimize import minimize_scalar
import time
from functools import partial
from utils import h
from profiler import profile_each_line

def h_B_plus_theta(coeffs, theta):
    """
    We don't sum h_B because it is constant and thus doesn't affect the minimum
    """
    return (
        coeffs[0] * theta**4
        + coeffs[1] * theta**3
        + coeffs[2] * theta**2
        + coeffs[3] * theta
    )


@profile_each_line
def tsd_trace_regression(X, Y, max_iters=100, verbose=True):
    N, n = X.shape
    B = np.identity(n)
    coord = [(i, j) for i in range(n) for j in range(i, n)]
    objectives = [h(X, Y, B)]
    times = []

    B_t_X = B.T @ X.T
    X_columns = [X[:, i] for i in range(n)]
    X_columns_squared = [X_columns[i]**2 for i in range(n)]
    X_columns_cubed = [X_columns_squared[i] * X_columns[i] for i in range(n)]
    X_columns_quadrupled = [X_columns_squared[i]**2 for i in range(n)]
    X_columns_quadrupled_sum = [np.sum(X_columns_quadrupled[i]) for i in range(n)]

    for t in range(max_iters):
        start = time.time()
        random.shuffle(coord)


        for i, j in coord:

            f_B_X = np.sum(B_t_X * B_t_X, axis=0)
            Y_minus_f = Y - f_B_X
            coeffs = [
                X_columns_quadrupled_sum[i],
                np.sum(4 * X_columns_cubed[i] * B_t_X[j]),
                np.sum(2*(2*B_t_X[j]**2 - 2 * Y_minus_f) * X_columns_squared[i]),
                -np.sum(4 * Y_minus_f * X_columns[i] * B_t_X[j]),
            ]
            min_funcion = partial(h_B_plus_theta, coeffs)
            theta = minimize_scalar(min_funcion).x

            B[i, j] += theta
            B_t_X[j] += theta * X_columns[i].T

        loop_time = time.time() - start

        h_B = h(X, Y, B)

        if verbose:
            print(f"Iteration: {t+1}, objective: {h_B}, time: {loop_time}")

        objectives.append(h_B)
        times.append(loop_time)

    return B, objectives, times
