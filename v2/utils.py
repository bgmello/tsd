import numpy as np

def objective(X, y, W):
    N = y.shape[0]
    return (1/(2*N))*np.sum([
        (y[p]-X[:, p].T@W@X[:, p])**2
        for p in range(N)
    ])