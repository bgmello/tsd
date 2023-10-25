import numpy as np
from scipy.linalg import expm

def rgd(X, y, r, max_iter=1000):
    n, N = X.shape
    U = np.column_stack([np.eye(n, r)])
    P = np.eye(r)
    ones_r = np.ones((r, 1))

    for t in range(max_iter):

        # Compute gradient
        GU = 2/N * np.sum([(np.dot(ones_r.T, (np.dot(U.T, X[:, [p]]) * np.dot(P, np.dot(U.T, X[:, [p]]))))[0, 0] - y[p]) * X[:, [p]] @ np.dot(P, np.dot(U.T, X[:, [p]])).T for p in range(N)], axis=0)
        A_tilda = np.dot(U.T, GU) @ P - P @ np.dot(U.T, GU)
        B_tilda = np.dot(U.T, GU)
        D_tilda = 0.5 * P @ np.dot(U.T, GU)

        # Exponential map update formula
        QR = GU - U @ np.dot(U.T, GU)
        Q, R = np.linalg.qr(QR)
        U_exp = np.column_stack([U, Q]) @ expm(np.block([[np.dot(U.T, GU) - np.dot(GU.T, U), -R.T], [R, np.zeros((r, r))]]))[:, :r]
        P_exp = P @ expm(0.5 * np.dot(U.T, GU))

        # Line search (placeholder, using a fixed step size for simplicity)
        gamma_star = 0.01

        # Update
        U = U_exp
        P = P_exp

    return U, P

# Sample test (using random data)
np.random.seed(0)
X = np.random.rand(5, 10)
y = np.random.rand(10)
U_final, P_final = rgd(X, y, r=2)

print(U_final, P_final)
