import numpy as np
from rgd import RGD
from v2.FRP.main import Algorithm
import json


if __name__ == "__main__":
    np.random.seed(42)
    n, N, r = 10, 500, 4

    # Inputs
    X = np.random.random((n, N)) # shape (n, N)
    y = np.random.random(N) # shape (N)

    algo = Algorithm(X, y, r)
    W, results_tsd = algo.run(num_iterations=200000, max_time=20)

    algo = RGD(X, y, r)
    W, results_rgd = algo.run(num_iterations=20000, max_time=20)

    with open("results.json", "w") as f:
        f.write(json.dumps({"tsd": results_tsd, "rgd": results_rgd}))