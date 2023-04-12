from tsd import tsd_trace_regression
import functools
from rgd import rgd_trace_regression
from multiprocessing.pool import ThreadPool

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
    iters = 1000

    def smap(f):
        return f()

    with ThreadPool(4) as pool:
        partials = [
                functools.partial(tsd_trace_regression, *generate_wishart(1000, 50, 50), max_iters=iters, verbose=True),
                functools.partial(rgd_trace_regression, *generate_wishart(1000, 50, 50), max_iters=iters),
                functools.partial(tsd_trace_regression, *generate_wishart(1000, 50, 10), max_iters=iters, verbose=True),
                functools.partial(rgd_trace_regression, *generate_wishart(1000, 50, 10), max_iters=iters)
                ]

        results = pool.map(smap, partials)
        pool.close()
        pool.join()

    pd.DataFrame({"iterations": np.arange(iters), "tsd": results[0][1], "rgd": results[1][1]}).to_csv("wishart_1000_50_50.csv", index=False)
    pd.DataFrame({"iterations": np.arange(iters), "tsd": results[2][1], "rgd": results[3][1]}).to_csv("wishart_1000_50_10.csv", index=False)
