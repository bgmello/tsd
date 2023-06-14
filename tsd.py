import numpy as np
from scipy.optimize import minimize_scalar
import time
from functools import partial
from utils import h


class TSDTraceRegression:
    def __init__(self):
        pass

    def fit(self, X, Y, tot_time, verbose=True, randomized=False):
        self.X = X
        self.Y = Y
        _, n = X.shape
        self.n = n 
        self.inner_time = time.time()
        self.B = np.identity(n)
        self.randomized = randomized
        self.objectives = [h(X, Y, self.B)]
        self.times = [0]
        self.inner_times = [0]
        self.inner_obj = [h(X, Y, self.B)]
        self.B_t_X = self.B.T @ X.T
        self.f_B_X = np.sum(self.B_t_X * self.B_t_X, axis=0)
        self.X_columns = [X[:, i] for i in range(n)]
        self.verbose = verbose

        self.coord = np.array([(i, j) for i in range(n) for j in range(i, n)])
        self.current_time = 0
        i = 0

        while True:
            self.run_iteration()
            i += 1
            if self.current_time > tot_time:
                break

        return self.B, self.objectives, self.times, self.inner_times, self.inner_obj
    
    def h_B_plus_theta(self, i, j, theta):
        theta_times_Xi = theta * self.X_columns[i]
        theta_squared_times_Xi2 = theta_times_Xi ** 2 
        temp = self.Y - self.f_B_X - 2 * theta_times_Xi * self.B_t_X[j] - theta_squared_times_Xi2
        return np.sum(temp ** 2)

    def find_best_theta(self, i, j, bounds=[None, None]):
        g = self.X_columns[i]
        h = self.B_t_X[j]
        f = self.Y - self.f_B_X
        coeffs = [
            np.sum(g**4),
            3*np.sum((g**3)*h),
            np.sum((g**2)*((2*(h**2))-f)),
            -np.sum(f*g*h)
        ]

        roots = np.roots(coeffs)

        min_val = np.inf  # Initialize with the max value
        best_theta = None

        if bounds[0] is not None:
            roots = list(roots)+[bounds[0]]

        for root in roots:
            if not np.isreal(root):
                continue

            if bounds[0] is not None and root.real < bounds[0]:
                continue

            if bounds[1] is not None and root.real > bounds[1]:
                continue

            theta = np.real(root)  # Extract the real part of the root

            val = self.h_B_plus_theta(i, j, theta)

            if val < min_val:  # If the current root gives a smaller h_B_plus_theta
                min_val = val
                best_theta = theta  # Update the best_theta

        return best_theta  # Return the theta that gives the smallest h_B_plus_theta

    def update_per_coord(self, pair):
        start = time.time()
        i, j = pair

        # find theta
        if i==j:
            theta = self.find_best_theta(i, j, bounds=[-self.B[i, j], None])
        else:
            theta = self.find_best_theta(i, j)

        new_obj = self.h_B_plus_theta(i, j, theta)

        # update B
        self.B[i, j] += theta

        # update values that depend on B
        self.f_B_X += (
            2 * theta * self.X_columns[i].T * self.B_t_X[j]
            + (theta * self.X_columns[i].T) ** 2
        )
        self.B_t_X[j] += theta * self.X_columns[i].T

        # store results
        time_inner_loop = time.time()-start
        self.inner_times.append(time.time()-self.inner_time)
        self.inner_obj.append(new_obj)
        self.inner_time = time.time()
        self.current_time += time_inner_loop


    def run_iteration(self):

        start = time.time()
        indices = np.random.choice(len(self.coord), size=len(self.coord), replace=self.randomized)
        list(map(self.update_per_coord, self.coord[indices]))

        h_B = h(self.X, self.Y, self.B)

        self.objectives.append(h_B)
        self.times.append(time.time()-start)