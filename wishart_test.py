from tsd import tsd_trace_regression
from rgd import rgd_trace_regression

import numpy as np
import pandas as pd


def generate_wishart(m, d, r):
    sigma = 1e-1

    X = np.zeros((m, d))
    Y = np.zeros(m)

    Sigmaorg = np.random.randn(d, r)
    Sigmaorg = Sigmaorg @ Sigmaorg.T

    for kk in range(m):
        X[kk] = np.random.randn(d)
        Y[kk] = np.dot(X[kk, :].T, np.dot(Sigmaorg, X[kk, :])) + sigma*np.random.randn()

    return X, Y


if __name__ == "__main__":
    # test 1
    X, Y = generate_wishart(1000, 50, 50)

    _, objectives_t = tsd_trace_regression(X, Y, max_iters=1000, verbose=False)
    _, objectives_r = rgd_trace_regression(X, Y, max_iters=1000)

    pd.DataFrame({"iterations": np.arange(1000), "tsd": objectives_t, "rgd": objectives_r}).to_csv("wishart_1000_50_50.csv", index=False)

    # test 2
    X, Y = generate_wishart(1000, 50, 10)

    _, objectives_t = tsd_trace_regression(X, Y, max_iters=1000, verbose=False)
    _, objectives_r = rgd_trace_regression(X, Y, max_iters=1000)

    pd.DataFrame({"iterations": np.arange(1000), "tsd": objectives_t, "rgd": objectives_r}).to_csv("wishart_1000_50_10.csv", index=False)
