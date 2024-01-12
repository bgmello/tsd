import numpy as np
from time import time
from quantities import Quantities
import k_loop
import ii_loop
import ij_loop


def run(quantities: Quantities, num_iterations=10, max_time=100, verbose=False):
    start = time()
    results = []
    print("Initial obj: {:.4}".format(quantities.objective()))

    for t in range(num_iterations):
        for k in range(quantities.r):
            continue
            obj = k_loop.loop(quantities, k)

            results.append([time() - start, obj])
            if verbose:
                print("Algo obj: {:.4}, Real obj: {:.4}, k: {}, time: {:.2}".format(obj, quantities.objective(), k, time()-start))

            if time() - start > max_time:
                return quantities.get_uput(), results

        for j in range(quantities.r):
            for i in range(j + 1):
                if i == j:
                    continue
                    obj = ii_loop.loop(quantities, i)
                else:
                    obj = ij_loop.loop(quantities, i, j)
                results.append([time() - start, obj])
                if time() - start > max_time:
                    return quantities.get_uput(), results

                if verbose:
                    print("Algo obj: {:.4}, Real obj: {:.4}, i-j: {}-{}, time: {:.2}s".format(obj, quantities.objective(), i, j, time()-start))

if __name__ == "__main__":
    # Inputs
    X = np.array([[1, 3], [2, 4]])
    y = np.array([6, 7])
    r = 2


    quantities = Quantities(X, y, r)
    run(quantities, num_iterations=5, verbose=True)
