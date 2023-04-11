import numpy as np
import random
from scipy.optimize import minimize_scalar
from utils import h

def tsd_trace_regression(X, Y, max_iters=100, verbose=True):
    """

    X.shape = (N, n)
    Y.shape = (N,)

    """

    objectives = []
    N, n = X.shape
    B = np.identity(n)  # shape (n, n)

    for t in range(max_iters):

        coord = [(i, j) for i in range(n) for j in range(i, n)]

        random.shuffle(coord)

        for (i, j) in coord:

            theta = minimize_scalar(lambda t: h(X, Y, B + (np.outer(np.arange(n) == i, np.arange(n) == j)) * theta)).x

            B[i, j] += theta

        print(f"Iteration: {t+1}, objective: {h(X, Y, B)}")
        objectives.append(h(X, Y, B))

    return B, objectives
