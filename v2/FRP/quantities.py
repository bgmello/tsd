import numpy as np

class Quantities:

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
        self.t = np.diag(self.X.T @ U @ self.P @ U.T @ self.X)
        assert np.all(
            [
                np.isclose(self.t[p], np.sum(self.X[:, p][: self.r] ** 2))
                for p in range(self.N)
            ]
        )

    def get_uput(self):
        return self.UQ @ np.diag(self.lambdas) @ self.UQ.T

    def objective(self):
        N = self.y.shape[0]
        W = self.get_uput()
        return (1/(2*N))*np.sum([
            (self.y[p]-self.X[:, p].T@W@self.X[:, p])**2
            for p in range(N)
        ])