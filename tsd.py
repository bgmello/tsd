import numpy as np
import random
from scipy.optimize import minimize_scalar
import time
from functools import partial

def h(f, Y):
    return np.sum((Y.flatten() - f)**2)

def f(B_t_v):
    return np.sum(B_t_v*B_t_v, axis=0)

def h_B_plus_theta(B_t_X_j, Y_minus_f, X_i, theta, h_B):
    w = 2 * theta * X_i * B_t_X_j + theta**2 * X_i**2
    return np.sum(h_B - 2*(Y_minus_f)*w + w**2)

def tsd_trace_regression(X, Y, max_iters=100, verbose=True):
    N, n = X.shape
    B = np.identity(n)
    coord = [(i, j) for i in range(n) for j in range(i, n)]
    objectives = [h(f(B.T@X.T), Y)]
    times = []
    minimize_times = []

    for t in range(max_iters):
        start = time.time()
        random.shuffle(coord)

        for (i, j) in coord:
            B_t_X = B.T @ X.T
            f_B_X = f(B_t_X)
            h_B = h(f_B_X, Y)
            X_i = X[:, i]
            Y_minus_f = Y - f_B_X
            B_t_X_j = B_t_X[j]

            min_funcion = partial(h_B_plus_theta, B_t_X_j, Y_minus_f, X_i, h_B=h_B)
            
            minimize_time = time.time()
            theta = minimize_scalar(min_funcion).x
            minimize_times.append(time.time() - minimize_time)

            B += theta * np.outer(np.eye(n)[i], np.eye(n)[j])

        loop_time = time.time() - start

        h_B = h(f(B.T@X.T), Y)

        if verbose:
            print(f"Iteration: {t+1}, objective: {h_B}, time: {loop_time}")

        objectives.append(h_B)
        times.append(loop_time)

    return B, objectives, times, minimize_times
