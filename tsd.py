import numpy as np
import time
from profiler import profile_each_line
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
        self.inner_time = time.process_time()
        self.B = np.identity(n)
        self.randomized = randomized
        self.objectives = [h(X, Y, self.B)]
        self.times = [0]
        self.inner_times = [0]
        self.inner_obj = [h(X, Y, self.B)]
        self.B_t_X = self.B.T @ X.T
        self.f_B_X = np.sum(self.B_t_X * self.B_t_X, axis=0)
        self.X_columns = [X[:, i] for i in range(n)]
        self.X_columns_squared = [self.X_columns[i]**2 for i in range(n)]
        self.X_columns_quadrupled_sum = [np.sum(self.X_columns_squared[i]**2) for i in range(n)]
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

    def h_B_plus_theta(self, Y_minus_f_B_X, i, j, theta):
        theta_times_Xi = theta * self.X_columns[i]
        theta_squared_times_Xi2 = theta_times_Xi ** 2 
        temp = Y_minus_f_B_X - 2 * theta_times_Xi * self.B_t_X[j] - theta_squared_times_Xi2
        return np.sum(temp ** 2)

    
    def find_best_theta(self, i, j, bounds=[None, None]):
        g = self.X_columns[i]
        g_squared = self.X_columns_squared[i]
        h = self.B_t_X[j]
        f = self.Y - self.f_B_X
        g_h = g*h
        coeffs = [
            self.X_columns_quadrupled_sum[i],
            3*np.sum(g_squared*g_h),
            np.sum((g_squared)*((2*(h**2))-f)),
            -np.sum(f*g_h)
        ]

        roots = np.roots(coeffs)

        min_val = np.inf  # Initialize with the max value
        best_theta = None

        if bounds[0] is not None:
            roots = list(roots)+[bounds[0]]

        h_B_plus_theta_i_j_fixed = partial(self.h_B_plus_theta, f, i, j)

        for root in roots:
            if not np.isreal(root):
                continue

            if bounds[0] is not None and root.real < bounds[0]:
                continue

            if bounds[1] is not None and root.real > bounds[1]:
                continue

            theta = np.real(root)  # Extract the real part of the root

            val = h_B_plus_theta_i_j_fixed(theta)

            if val < min_val:  # If the current root gives a smaller h_B_plus_theta
                min_val = val
                best_theta = theta  # Update the best_theta

        return best_theta, min_val  # Return the theta that gives the smallest h_B_plus_theta

    def update_per_coord(self, pair):
        start = time.process_time()
        i, j = pair

        # find theta
        if i==j:
            theta, new_obj = self.find_best_theta(i, j, bounds=[-self.B[i, j], None])
        else:
            theta, new_obj = self.find_best_theta(i, j)

        
        # update B
        self.B[i, j] += theta
        
        # update values that depend on B
        theta_X_i_transpose = theta * self.X_columns[i].T
        self.f_B_X += (
            2 * theta_X_i_transpose * self.B_t_X[j]
            + (theta_X_i_transpose) ** 2
        )
        self.B_t_X[j] += theta_X_i_transpose

        # store results
        time_inner_loop = time.process_time()-start
        self.inner_times.append(time.process_time()-self.inner_time)
        self.inner_obj.append(new_obj)
        self.inner_time = time.process_time()
        self.current_time += time_inner_loop


    def run_iteration(self):

        start = time.process_time()
        indices = np.random.choice(len(self.coord), size=len(self.coord), replace=self.randomized)
        list(map(self.update_per_coord, self.coord[indices]))

        h_B = h(self.X, self.Y, self.B)

        self.objectives.append(h_B)
        self.times.append(time.process_time()-start)