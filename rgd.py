import numpy as np
from scipy.optimize import minimize_scalar
from utils import h
import time


def B_eta(B, G, eta):
    n = B.shape[0]
    B_eta = B.copy()
    for i in range(n):
        for j in range(i + 1):
            if i == j:
                B_eta[i, j] += B[i, i] * (np.exp(-eta * B[i, i] * G[i, j]) - 1)
            else:
                B_eta[i, j] -= eta * G[i, j]
    return B_eta


def rgd_trace_regression(X, Y, tot_time):
    N, n = X.shape
    B = np.identity(n)
    objectives = [h(X, Y, B)]
    times = [0]
    current_time = 0
    i = 0

    while True:
        i += 1
        start = time.time()
        B_prev = B.copy()
        V = B_prev.T @ X.T
        F = np.sum(V * V, axis=0)

        G = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1):
                G[i, j] = -4 * np.sum((Y.flatten() - F) * X[:, i] * V[j])

        eta_t = minimize_scalar(lambda eta: h(X, Y, B_eta(B, G, eta))).x
        B = B_eta(B, G, eta_t)
        objectives.append(h(X, Y, B))
        times.append(time.time() - start)
        current_time += times[-1]

        if current_time > tot_time:
            break

    return B, objectives, times
