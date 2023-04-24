from tsd import TSDTraceRegression
import json
from rgd import rgd_trace_regression

import numpy as np

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
    tot_time = 20

    test_data = [
        (1000, 50, 50),
        (1000, 50, 10)
    ]

    for params in test_data:
        X, Y = generate_wishart(*params)
        _, objectives_t, times_t, inner_times = TSDTraceRegression().fit(X, Y, tot_time, verbose=True)
        _, objectives_t_rand, times_t_rand, inner_times_rand = TSDTraceRegression().fit(X, Y, tot_time, verbose=True, randomized=True)
        _, objectives_r, times_r = rgd_trace_regression(X, Y, tot_time)
        with open(f"wishart_{'_'.join([str(param) for param in params])}.json", "w") as f:
            f.write(json.dumps({"max_time": tot_time,
                                "tsd_objective": objectives_t,
                                "tsd_objective_rand": objectives_t_rand,
                                "rgd_objective": objectives_r,
                                "tsd_time": times_t,
                                "tsd_time_rand": times_t_rand,
                                "tsd_inner_time": inner_times,
                                "tsd_inner_time_rand": inner_times_rand,
                                "rgd_time": times_r}))
