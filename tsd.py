import numpy as np
import random
from scipy.optimize import minimize_scalar
from utils import h
import time

def w_ij(B, X, theta, i, j):
    B_t_v = B.T @ X.T
    e_i_t_v = X[:, i]
    return 2 * theta * e_i_t_v * B_t_v[j] + theta**2 * e_i_t_v**2

def f(B, X):
    V = B.T @ X.T
    return np.sum(V*V, axis=0)

def h_B_plus_theta(B, X, Y, theta, i, j, h_B, f):
    w = w_ij(B, X, theta, i, j)
    return np.sum(h_B - 2*(Y-f)*w + w**2)

def tsd_trace_regression(X, Y, max_iters=100, verbose=True):
    """

    X.shape = (N, n)
    Y.shape = (N,)

    """

    objectives = []
    times = []
    N, n = X.shape
    B = np.identity(n)  # shape (n, n)

    coord = [(i, j) for i in range(n) for j in range(i, n)]

    objectives.append(h(X, Y, B))

    for t in range(max_iters):

        start = time.time()

        random.shuffle(coord)

        for (i, j) in coord:

            h_B = h(X, Y, B)
            f_B_X = f(B, X)

            theta = minimize_scalar(lambda t: h_B_plus_theta(B, X, Y, t, i, j, h_B, f_B_X)).x

            B[i, j] += theta

        loop_time = time.time() - start

        if verbose:
            print(f"Iteration: {t+1}, objective: {h(X, Y, B)}, time: {loop_time}")
        objectives.append(h(X, Y, B))
        times.append(loop_time)

    return B, objectives, times