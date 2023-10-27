import numpy as np
from scipy.optimize import minimize_scalar
from time import time
from utils import objective
from profiler import profile_each_line


class Algorithm:
    def __init__(self, X, y, r):
        self.r = r
        self.n = X.shape[0]
        self.N = X.shape[1]
        self.X, self.y = X, y

        assert X.shape[1] == y.shape[0]
        assert self.r <= self.n

        U = np.eye(self.n, r)

        self.P = np.eye(self.r)  # shape (r,r)
        self.UQ = np.eye(self.n, r)
        self.lambdas = np.ones(self.r)  # shape (r)
        self.W = self.X.T @ np.eye(self.n, r)  # shape (N, r)
        self.t = np.array(
            [
                self.X[:, p].T @ U @ self.P @ U.T @ self.X[:, p]
                for p in range(self.N)
            ]
        )
        assert np.all(
            [
                np.isclose(self.t[p], np.sum(self.X[:, p][: self.r] ** 2))
                for p in range(self.N)
            ] 
        )

    def inner_loop_k_subspace(self, k):
        v_k = self.v_k(k)
        optimal_gamma = self.find_optimal_k_gamma(v_k, k)

        s = self.s(v_k, k, optimal_gamma)
        z = self.z(v_k, k, optimal_gamma)
        
        self.W[:, k] = self.X.T @ (s + self.UQ[:, k])  # X.T shape = N, n
        self.t += 2 * self.lambdas[k] * self.W[:, k] * z - self.lambdas[k] * (z * z)
        self.UQ[:, k] += s

    def run(self, num_iterations=10, max_time=100):
        start = time()

        for t in range(num_iterations):
            # for k in range(self.r):
            #     self.inner_loop_k_subspace(k)

            #     if time() - start > max_time:
            #         return self.get_uput()

            for j in range(self.r):
                for i in range(j+1):
                    self.inner_loop_ij_subspace(i, j)
                    print("ij {}{} obj {} ".format(i, j, objective(self.X, self.y, self.get_uput())))
                    print(self.h_i_j(i, j, 0))
                    if time() - start > max_time:
                        return self.get_uput()

    # @profile_each_line
    def v_k(self, k):
        t_minus_y_W_k = (self.t - self.y) * self.W[:, k]
        v_hat = (2 * self.lambdas[k] / self.N) * (
            self.X @ (t_minus_y_W_k)
            - self.UQ @ (self.W.T @ (t_minus_y_W_k))
        )
        if np.linalg.norm(v_hat, ord=2) == 0:
            return np.zeros_like(v_hat)
        return v_hat / np.linalg.norm(v_hat, ord=2)

    # @profile_each_line
    def h(self, v_k, k, gamma):
        z = self.z(v_k, k, gamma)
        lambda_z = self.lambdas[k] * z
        res = self.y - self.t - 2 * self.W[:, k] * lambda_z - lambda_z *  z
        return (1 / (2 * self.N)) * np.linalg.norm(res, ord=2)

    def find_optimal_k_gamma(self, v_k, k):
        objective = lambda gamma: self.h(v_k, k, gamma)
        result = minimize_scalar(objective, bounds=(0, 2 * np.pi), method="bounded")

        return result.x

    def S_i_j(self, i, j, gamma):
        alpha_i_j = np.sqrt(self.lambdas[i]*self.lambdas[j])*(1/(self.lambdas[i])-1/(self.lambdas[j]))
        cos = np.cos(alpha_i_j*gamma)
        sin = np.sqrt(1-cos**2)
        return np.array([
            [cos, sin],
            [-sin, cos]])

    def T_i_j(self, i, j, gamma):
        lambda_cos_i = self.lambdas[i]*np.cosh(gamma/2)
        lambda_cos_j = self.lambdas[j]*np.cosh(gamma/2)
        lambda_sin = np.sqrt(self.lambdas[i]*self.lambdas[j])*np.sinh(gamma/2)
        return np.array([
            [lambda_cos_i, lambda_sin],
            [lambda_sin, lambda_cos_j]])

    # @profile_each_line
    def V_i_j(self, i, j, gamma):
        s = self.S_i_j(i, j, gamma)
        t = self.T_i_j(i, j, gamma)

        return s @ t @ s.T

    def v_i_j(self, i, j, gamma):
        v_i_j = (
            self.t 
            - self.lambdas[i] * (self.W[:, i]*self.W[:, i])
            - self.lambdas[j] * (self.W[:, j]*self.W[:, j])
            + np.sum(((self.W[:, [i, j]] @ self.V_i_j(i, j, gamma)) * self.W[:, [i, j]]), axis=1)
            )

        return v_i_j

    def h_i_j(self, i, j, gamma):
        return (1/(2*self.N))*np.linalg.norm(self.y-self.v_i_j(i, j, gamma), ord=2)**2


    def inner_loop_ij_subspace(self, i, j):
        objective = lambda gamma: self.h_i_j(i, j, gamma)

        bounds = ([-1, 0]
        if 2*np.sqrt(self.lambdas[i]*self.lambdas[j])*(self.t-self.y).T @ (self.W[:, i] * self.W[:, j])/self.N > 0
        else [0, 1])

        result = minimize_scalar(objective, bounds=bounds, method='bounded')

        optimal_gamma = result.x
        if np.isnan(result.fun):
            exit()

        self.t = self.v_i_j(i, j, optimal_gamma)

        eigenvalues, eigenvectors = np.linalg.eig(self.T_i_j(i, j, optimal_gamma))

        # Extract the values
        v_i_plus = eigenvectors[:, 0]
        v_j_plus = eigenvectors[:, 1]

        self.W[:, i] = self.W[:, [i, j]] @ v_i_plus
        self.W[:, j] = self.W[:, [i, j]] @ v_j_plus
        self.UQ[:, i] = self.UQ[:, [i, j]] @ v_i_plus
        self.UQ[:, j] = self.UQ[:, [i, j]] @ v_j_plus


    def s(self, v_k, k, gamma):
        cos = np.cos(gamma)
        sin = np.sqrt(1-cos**2)
        return (cos - 1) * self.UQ[:, k] + sin * v_k

    # @profile_each_line
    def z(self, v_k, k, gamma):
        cos = np.cos(gamma)
        sin = np.sqrt(1-cos**2)
        return ((cos - 1) * self.W[:, k] + sin * self.X.T @ v_k)

    def get_uput(self):
        return self.UQ @ np.diag(self.lambdas) @ self.UQ.T


if __name__ == "__main__":
    np.random.seed(42)
    n, N, r = 7, 10, 5

    # Inputs
    X = np.random.random((n, N))  # shape (n, N)
    y = np.zeros(N)  # shape (N)

    algo = Algorithm(X, y, r)
    algo.run(num_iterations=2000)
