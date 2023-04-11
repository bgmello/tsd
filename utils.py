import numpy as np


def h(X, Y, B):
    V = B.T @ X.T
    return np.sum((Y.flatten() - np.sum(V * V, axis=0))**2)
