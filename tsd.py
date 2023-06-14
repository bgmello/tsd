import numpy as np
from scipy.optimize import minimize_scalar
import time
from functools import partial
from utils import h
from profiler import profile_each_line


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
        self.X_columns_squared = [self.X_columns[i] ** 2 for i in range(n)]
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
    
    def h_B_plus_theta(self, y_minus_f_B_X, i, j, theta):
        theta_times_Xi = theta * self.X_columns[i]
        theta_squared_times_Xi2 = theta_times_Xi ** 2 
        temp = y_minus_f_B_X - 2 * theta_times_Xi * self.B_t_X[j] - theta_squared_times_Xi2
        return np.sum(temp ** 2)

    def update_per_coord(self, pair):
        start = time.time()
        i, j = pair

        # find theta
        min_function = partial(self.h_B_plus_theta, self.Y-self.f_B_X, i, j)
        results = minimize_scalar(min_function)
        theta, new_obj = results.x, results.fun

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