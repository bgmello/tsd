from tsd import TSDTraceRegression
from rgd import rgd_trace_regression

import numpy as np
import pandas as pd

np.random.seed(42)


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
    max_iters = 30

    test_data = [
        (1000, 50, 50),
        (1000, 50, 10)
    ]

    for params in test_data:
        X, Y = generate_wishart(*params)
        _, objectives_t, times_t = TSDTraceRegression().fit(X, Y, max_iters, verbose=True)
        _, objectives_r, times_r = rgd_trace_regression(X, Y, max_iters)
        pd.DataFrame({"iterations": np.arange(max_iters+1),
                      "tsd_objective": objectives_t,
                      "rgd_objective": objectives_r,
                      "tsd_time": times_t,
                      "rgd_time": times_r}).to_csv(f"wishart_{'_'.join([str(param) for param in params])}.csv", index=False)
