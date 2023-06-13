from tsd import TSDTraceRegression
import json
from rgd import rgd_trace_regression
import sys

from multiprocessing import Pool
import numpy as np

num_seeds = int(sys.argv[2])
tot_time = int(sys.argv[1])


def generate_wishart(m, d, r, seed):
    sigma = 1e-1
    np.random.seed(seed)

    X = np.zeros((m, d))
    Y = np.zeros(m)

    Sigmaorg = np.random.randn(d, r)
    Sigmaorg = Sigmaorg @ Sigmaorg.T

    for kk in range(m):
        X[kk] = np.random.randn(d)
        Y[kk] = np.dot(X[kk, :].T, np.dot(Sigmaorg, X[kk, :])) + sigma*np.random.randn()

    return X, Y


def generate_data(input_):

    m, d, r, seed, algo = input_

    X, Y = generate_wishart(m, d, r, seed)
    if algo == "tsd":
        _, objectives_t, times_t, inner_times, inner_obj = TSDTraceRegression().fit(X, Y, tot_time, verbose=True)
        with open(f"data/wishart_{m}_{d}_{r}_seed_{seed}_algo_{algo}.json", "w") as f:
            f.write(json.dumps({"max_time": tot_time,
                                "tsd_objective": objectives_t,
                                "tsd_inner_objective": inner_obj,
                                "tsd_time": times_t,
                                "tsd_inner_time": inner_times}))
    if algo == "tsd_rand":
        _, objectives_t_rand, times_t_rand, inner_times_rand, inner_obj_rand = TSDTraceRegression().fit(X, Y, tot_time, verbose=True, randomized=True)
        with open(f"data/wishart_{m}_{d}_{r}_seed_{seed}_algo_{algo}.json", "w") as f:
            f.write(json.dumps({"tsd_objective_rand": objectives_t_rand,
                                "tsd_inner_objective_rand": inner_obj_rand,
                                "tsd_time_rand": times_t_rand,
                                "tsd_inner_time_rand": inner_times_rand,
                                }))
    if algo == "rgd":
        _, objectives_r, times_r = rgd_trace_regression(X, Y, tot_time)
        with open(f"data/wishart_{m}_{d}_{r}_seed_{seed}_algo_{algo}.json", "w") as f:
            f.write(json.dumps({"rgd_objective": objectives_r,
                                "rgd_time": times_r}))


if __name__ == "__main__":

    ms = [1000]
    ds = [50]
    rs = [10, 50]

    inputs = []
    for m in ms:
        for d in ds:
            for r in rs:
                for seed in range(num_seeds):
                    for algo in ["tsd", "tsd_rand", "rgd"]:
                        inputs.append((m, d, r, seed, algo))

    with Pool() as p:
        p.map(generate_data, inputs)
