import numpy as np
from rgd import RGD
from fixed_rank_psd import Algorithm
from time import time
import helper


if __name__ == "__main__":
    np.random.seed(42)
    n, N, r = 10, 50000, 4

    # Inputs
    X = np.random.random((n, N)) # shape (n, N)
    y = np.random.random(N) # shape (N)

    start = time()
    algo = RGD(X, y, r)
    algo.run(num_iterations=20000, max_time=20)
    print("RGD time ", time()-start)

    start = time()
    algo = Algorithm(X, y, r)
    algo.run(num_iterations=200000, max_time=20)
    print("Algo time ", time()-start)