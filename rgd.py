import numpy as np
from scipy.optimize import minimize_scalar
from utils import h


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


def rgd_trace_regression(X, Y, max_iters=100):
    N, n = X.shape
    B = np.identity(n)
    objectives = []

    for t in range(max_iters):
        B_prev = B.copy()
        V = B_prev.T @ X.T
        F = np.sum(V * V, axis=0)

        G = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1):
                G[i, j] = -4 * np.sum((Y.flatten() - F) * X[:, i] * V[j])

        eta_t = minimize_scalar(lambda eta: h(X, Y, B_eta(B, G, eta))).x
        B = B_eta(eta_t)
        objectives.append(h(B))

    return B, objectives
