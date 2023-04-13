import numpy as np
import random
from scipy.optimize import minimize_scalar
import time
from functools import partial
from utils import h

class TSDTraceRegression:
    def __init__(self):
        pass

    def fit(self, X, Y, max_iters=100, verbose=True):
        self.X = X
        self.Y = Y
        N, n = X.shape
        self.B = np.identity(n)
        self.objectives = [h(X, Y, self.B)]
        self.times = [np.nan]
        self.B_t_X = self.B.T @ X.T
        self.f_B_X = np.sum(self.B_t_X * self.B_t_X, axis=0)
        self.X_columns = [X[:, i] for i in range(n)]
        self.X_columns_squared = [self.X_columns[i] ** 2 for i in range(n)]
        self.X_columns_cubed = [
            self.X_columns_squared[i] * self.X_columns[i] for i in range(n)
        ]
        self.X_columns_quadrupled = [self.X_columns_squared[i] ** 2 for i in range(n)]
        self.X_columns_quadrupled_sum = [
            np.sum(self.X_columns_quadrupled[i]) for i in range(n)
        ]

        coord = [(i, j) for i in range(n) for j in range(i, n)]
        objectives = [h(X, Y, self.B)]
        times = [np.nan]

        for t in range(max_iters):
            start = time.time()
            random.shuffle(coord)
            loop_time = time.time() - start

            list(map(self.update_per_coord, coord))

            h_B = h(X, Y, self.B)

            if verbose:
                print(f"Iteration: {t+1}, objective: {h_B}, time: {loop_time}")

            objectives.append(h_B)
            times.append(loop_time)

        return self.B, objectives, times

    @staticmethod
    def h_B_plus_theta(coeffs, theta):
        """
        We don't sum h_B because it is constant and thus doesn't affect the minimum
        """
        return (
            coeffs[0] * theta**4
            + coeffs[1] * theta**3
            + coeffs[2] * theta**2
            + coeffs[3] * theta
        )

    def update_per_coord(self, pair):
        i, j = pair
        Y_minus_f = self.Y - self.f_B_X
        coeffs = [
            self.X_columns_quadrupled_sum[i],
            np.sum(4 * self.X_columns_cubed[i] * self.B_t_X[j]),
            np.sum(
                2 * (2 * self.B_t_X[j] ** 2 - 2 * Y_minus_f) * self.X_columns_squared[i]
            ),
            -np.sum(4 * Y_minus_f * self.X_columns[i] * self.B_t_X[j]),
        ]
        min_funcion = partial(self.h_B_plus_theta, coeffs)
        theta = minimize_scalar(min_funcion).x

        self.B[i, j] += theta
        self.f_B_X += (
            2 * theta * self.X_columns[i].T * self.B_t_X[j]
            + (theta * self.X_columns[i].T) ** 2
        )
        self.B_t_X[j] += theta * self.X_columns[i].T
