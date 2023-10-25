import numpy as np
from scipy.optimize import minimize_scalar


class Algorithm:
    def __init__(self, X, y, r):
        self.r = r
        self.n = X.shape[0]
        self.N = X.shape[1]
        self.X, self.y = X, y

        assert X.shape[1] == y.shape[0]
        assert self.r <= self.n

        self.U = np.concatenate([self.e(i, self.n).reshape(-1, 1) for i in range(self.r)], axis=-1)  # shape(n, r)

        self.P = np.eye(self.r)  # shape (r,r)
        self.UQ = np.concatenate([self.e(i, self.n).reshape(-1, 1) for i in range(self.r)], axis=-1)  # shape(n, r)
        self.lambdas = np.ones(self.r)  # shape (r)
        self.W = self.X.T @ np.concatenate([self.e(i, self.n).reshape(-1, 1) for i in range(self.r)], axis=-1)  # shape (N, r)
        self.t = np.array([self.X[:, p].T @ self.U @ self.P @ self.U.T @ self.X[:, p] for p in range(self.N)])
        assert (np.all([np.isclose(self.t[p], np.sum(self.X[:, p][:self.r]**2)) for p in range(self.N)]))

    def inner_loop_k_subspace(self, k):
        optimal_gamma = self.find_optimal_k_gamma(k)

        s = self.s(k, optimal_gamma)
        z = self.z(k, optimal_gamma)
        print("h: ", self.h(k, optimal_gamma))

        self.W[:, k] = self.X.T @ (s + self.UQ[:, k])  # X.T shape = N, n
        self.t += 2*self.lambdas[k]*self.W[:, k]*z - self.lambdas[k]*(z*z)
        self.UQ[:, k] += s

    def inner_loop_ij_subspace(self, i, j):
        pass

    def run(self, num_iterations=10):

        for t in range(num_iterations):
            for k in range(self.r):
                self.inner_loop_k_subspace(k)

    def v_k(self, k):
        v_hat = (2*self.lambdas[k]/self.N)*(self.X @ ((self.t-self.y)*self.W[:, k]) - self.UQ @ (self.W.T @ ((self.t-self.y)*self.W[:, k])))
        return v_hat/np.linalg.norm(v_hat, ord=2)

    def v_ij(self, i, j):
        pass


    def h(self, k, gamma):
        z = self.z(k, gamma)
        return (1/(2*self.N))*np.linalg.norm(self.y-self.t-2*self.lambdas[k]*self.W[:, k]
                                             * z - self.lambdas[k]*(z*z), ord=2)

    def find_optimal_k_gamma(self, k):
        objective = lambda gamma: self.h(k, gamma)
        result = minimize_scalar(objective, bounds=(0, 2*np.pi), method='bounded')
    
        return result.x

    def find_optimal_ij_gamma(self, i, j):
        pass

    def s(self, k, gamma):
        return (np.cos(gamma)-1)*self.UQ[:, k] + np.sin(gamma)*self.v_k(k)

    def z(self, k, gamma):
        return (np.cos(gamma)-1)*self.W[:, k]+np.sin(gamma) * self.X.T @ self.v_k(k)

    @staticmethod
    def e(k, r):
        if k > r:
            raise ValueError("k should be less than or equal to r")

        # Create a zero vector of size r
        e_kr = np.zeros(r)

        # Set the k-th entry to 1
        e_kr[k] = 1

        return e_kr


if __name__ == "__main__":
    n, N, r = 7, 10, 4

    # Inputs
    X = np.random.random((n, N)) # shape (n, N)
    y = np.zeros(N) # shape (N)

    algo = Algorithm(X, y, r)
    algo.run(num_iterations=2000)

