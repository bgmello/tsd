from tsd import TSDTraceRegression
import json
from rgd import rgd_trace_regression
import sys

from multiprocessing import Pool
from multiprocessing import Manager
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

    m, d, r, seed = input_

    X, Y = generate_wishart(m, d, r, seed)
    _, objectives_t, times_t, inner_times = TSDTraceRegression().fit(X, Y, tot_time, verbose=True)
    _, objectives_t_rand, times_t_rand, inner_times_rand = TSDTraceRegression().fit(X, Y, tot_time, verbose=True, randomized=True)
    _, objectives_r, times_r = rgd_trace_regression(X, Y, tot_time)
    results[(m,d,r)].append({"max_time": tot_time,
                            "tsd_objective": objectives_t,
                            "tsd_objective_rand": objectives_t_rand,
                            "rgd_objective": objectives_r,
                            "tsd_time": times_t,
                            "tsd_time_rand": times_t_rand,
                            "tsd_inner_time": inner_times,
                            "tsd_inner_time_rand": inner_times_rand,
                            "rgd_time": times_r})

if __name__ == "__main__":

    with Manager() as manager:

        results = manager.dict()

        ms = [1000, 10000]
        ds = [50, 100, 500, 1000]
        rs = [10, 50]

        inputs = []
        for m in ms:
            for d in ds:
                for r in rs:
                    results[(m,d,r)] = []
                    for seed in range(num_seeds):
                        inputs.append((m,d,r, seed))


        with Pool() as p:
            p.map(generate_data, inputs)
        
        for m in ms:
            for d in ds:
                for r in rs:
                    with open(f"data/wishart_{m}_{d}_{r}.json", "w") as f:
                        f.write(json.dumps(results[(m,d,r)]))
