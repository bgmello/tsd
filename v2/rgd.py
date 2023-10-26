import numpy as np
from scipy.linalg import expm, qr
from scipy.optimize import minimize_scalar
from time import time

class RGD:
    def __init__(self, X, y, r):
        self.r = r
        self.n = X.shape[0]
        self.N = X.shape[1]
        self.X, self.y = X, y
        self.U = np.eye(self.n, self.r)
        self.P = np.eye(r)  # Initial P
    
        self.one_r = np.ones(r)

        assert X.shape[1] == y.shape[0]
        assert self.r <= self.n
    
    def G(self):
        """
        Compute the Euclidean gradient G.
        
        Parameters:
        - X: Matrix of features, shape (n, N).
        - U: Current U matrix, shape (n, r).
        - P: Current P matrix, shape (r, r).
        - y: Vector of responses, shape (N,).
        - N: Number of samples.
        
        Returns:
        - G: Euclidean gradient, shape (n, n).
        """
        # Initialize gradient
        G = np.zeros((self.n, self.n))

        # Loop over each sample and update G
        for p in range(self.N):
            x_p = self.X[:, p].reshape(-1, 1)
            G += (x_p.T @ self.U @ self.P @ self.U.T @ x_p - y[p]) * x_p @ x_p.T

        G /= self.N

        return G

    def compute_U_plus_P_plus(self, gamma):
        """
        Compute the U_plus and P_plus matrices for a given gamma.
        
        Parameters:
        - U: Current U matrix, shape (n, r).
        - G: Euclidean gradient, shape (n, n).
        - P: Current P matrix, shape (r, r).
        - A, B, D: Matrices as defined previously.
        - gamma: Step size.
        
        Returns:
        - U_plus, P_plus: Updated U and P matrices.
        """

        G = np.zeros((self.n, self.n))

        # Loop over each sample and update G
        for p in range(self.N):
            x_p = self.X[:, p].reshape(-1, 1)
            G += (x_p.T @ self.U @ self.P @ self.U.T @ x_p - self.y[p]) * x_p @ x_p.T

        G /= self.N

        # Compute A, B, D
        A = self.U.T @ G @ self.U @ self.P - self.P @ self.U.T @ G @ self.U
        B = (self.U.T @ G @ self.U).T @ self.P
        D = 0.5 * self.P @ self.U.T @ G @ self.U @ self.P
        # Compute GU
        GU = (G + G.T) @ self.U @ self.P
        
        # Compute QR decomposition
        Q, R = qr(GU - self.U @ (self.U.T @ GU))
        
        # Exponential map for U
        block_matrix = np.block([[self.U.T @ GU - GU.T @ self.U, -R.T], [R, np.zeros((R.shape[0], R.shape[0]))]])
        exp_block = expm(gamma * block_matrix)
        
        U_plus = np.hstack([self.U, Q]) @ exp_block[:, :self.U.shape[1]]
        
        # Exponential map for P
        P_plus = self.P @ expm((gamma / 2) * self.U.T @ GU)

        return U_plus, P_plus

    def h(self, gamma):
        """
        Compute the h function for a given U_plus and P_plus.
        
        Parameters:
        - U_plus: Updated U matrix, shape (n, r).
        - P_plus: Updated P matrix, shape (r, r).
        - X: Matrix of features, shape (n, N).
        - y: Vector of responses, shape (N,).
        - N: Number of samples.
        
        Returns:
        - h_value: Value of the h function.
        """
        # Compute tau_plus
        U_plus, P_plus = self.compute_U_plus_P_plus(gamma)
        U_plus_X = U_plus.T @ self.X
        P_plus_U_plus_X = P_plus @ U_plus_X
        tau_plus = np.sum(U_plus_X * P_plus_U_plus_X, axis=0)
        
        # Compute h
        h_value = (1 / (2 * self.N)) * np.linalg.norm(self.y - tau_plus)**2
        
        return h_value

    def run(self, num_iterations=20, max_time=100):

        start = time()

        for _ in range(num_iterations):
            result = minimize_scalar(self.h)

            optimal_gamma = result.x

            self.U, self.P = self.compute_U_plus_P_plus(optimal_gamma)

            if time()-start>=max_time:
                break

        print(self.h(0))
        return self.U, self.P

# Test with some random data
if __name__ == "__main__":
    n, N, r = 7, 10, 4

    # Inputs
    X = np.random.random((n, N)) # shape (n, N)
    y = np.random.random(N) # shape (N)

    algo = RGD(X, y, r)
    algo.run(num_iterations=2000)