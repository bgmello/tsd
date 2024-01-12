import numpy as np
from scipy.optimize import minimize_scalar


def S(quantities, i, j, gamma):
    alpha_i_j = np.sqrt(quantities.lambdas[i] * quantities.lambdas[j]) * (
        np.power(quantities.lambdas[i], -1) - np.power(quantities.lambdas[j], -1)
    )
    cos = np.cos(alpha_i_j * gamma)
    sin = np.sin(alpha_i_j * gamma)
    return np.array([[cos, sin], [-sin, cos]])


def T(quantities, i, j, gamma):
    lambda_cos_i = quantities.lambdas[i] * np.cosh(gamma / 2)
    lambda_cos_j = quantities.lambdas[j] * np.cosh(gamma / 2)
    lambda_sin = np.sqrt(quantities.lambdas[i] * quantities.lambdas[j]) * np.sinh(gamma / 2)
    return np.array([[lambda_cos_i, lambda_sin], [lambda_sin, lambda_cos_j]])


def v(quantities, i, j, gamma):
    s = S(quantities, i, j, gamma)
    t = T(quantities, i, j, gamma)

    V = s @ t @ s.T
    w_i = quantities.W[:, i]
    w_j = quantities.W[:, j]

    w_i_w_j = np.column_stack((w_i, w_j))

    return (
        quantities.t
        - quantities.lambdas[i] * (w_i * w_i)
        - quantities.lambdas[j] * (w_j * w_j)
        + np.sum(
            ((w_i_w_j @ V) * w_i_w_j),
            axis=1,
        )
    )


def h(quantities, i, j, gamma):
    return (1 / (2 * quantities.N)) * np.linalg.norm(
        quantities.y - v(quantities, i, j, gamma), ord=2
    ) ** 2

def get_eigen_values_and_vectors(quantities, i, j, gamma):
    alpha = gamma/2
    lambda_1 = (np.cosh(alpha)/2)*(quantities.lambdas[i]+quantities.lambdas[j]-np.sqrt((quantities.lambdas[i]-quantities.lambdas[j])**2+4*(np.tanh(alpha)**2)*(quantities.lambdas[i]*quantities.lambdas[j])))
    lambda_2 = (np.cosh(alpha)/2)*(quantities.lambdas[i]+quantities.lambdas[j]+np.sqrt((quantities.lambdas[i]-quantities.lambdas[j])**2+4*(np.tanh(alpha)**2)*(quantities.lambdas[i]*quantities.lambdas[j])))

    v_1_unnormalized = np.array([
        quantities.lambdas[i]-quantities.lambdas[j]-np.sqrt((quantities.lambdas[i]-quantities.lambdas[j])**2+4*(np.tanh(alpha)**2)*(quantities.lambdas[i]*quantities.lambdas[j])),
        2*np.tanh(alpha)*np.sqrt(quantities.lambdas[i]*quantities.lambdas[j])
    ])

    v_2_unnormalized = np.array([
        quantities.lambdas[i]-quantities.lambdas[j]+np.sqrt((quantities.lambdas[i]-quantities.lambdas[j])**2+4*(np.tanh(alpha)**2)*(quantities.lambdas[i]*quantities.lambdas[j])),
        2*np.tanh(alpha)*np.sqrt(quantities.lambdas[i]*quantities.lambdas[j])
    ])

    print("Unnormalized: ", v_1_unnormalized)
    print("Unnormalized: ", v_2_unnormalized)

    v1 = v_1_unnormalized/np.linalg.norm(v_1_unnormalized, ord=2)
    v2 = v_2_unnormalized/np.linalg.norm(v_2_unnormalized, ord=2)

    return lambda_1, lambda_2, v1, v2

def loop(quantities, i, j):
    objective = lambda gamma: h(quantities, i, j, gamma)

    bounds = (
        [-20*np.pi, 0]
        if 2
        * np.sqrt(quantities.lambdas[i] * quantities.lambdas[j])
        * (quantities.t - quantities.y).T
        @ (quantities.W[:, i] * quantities.W[:, j])
        / quantities.N
        > 0
        else [0, 20*np.pi]
    )

    result = minimize_scalar(objective, bounds=bounds, method="bounded")

    optimal_gamma = result.x
    print("Gamma: ", optimal_gamma)

    quantities.t = v(quantities, i, j, optimal_gamma)

    print("T: ", T(quantities, i, j, optimal_gamma))
    print("Lambda i: ", quantities.lambdas[i])
    print("Lambda j: ", quantities.lambdas[j])

    eigenvalues, eigenvectors = np.linalg.eig(T(quantities, i, j, optimal_gamma))

    # Extract the values
    # v_i_plus = eigenvectors[:, 0]
    # v_j_plus = eigenvectors[:, 1]
    # lambda_i_plus = eigenvalues[0]
    # lambda_j_plus = eigenvalues[1]
    lambda_i_plus, lambda_j_plus, v_i_plus, v_j_plus = get_eigen_values_and_vectors(quantities, i, j, optimal_gamma)

    print("v i from appendix: ", v_i_plus)
    print("v j from appendix: ", v_j_plus)
    print("Eigenvectors computed using numpy:")
    print("i: ", eigenvectors[:, 0])
    print("j: ", eigenvectors[:, 1])

    quantities.W[:, i] = quantities.W[:, [i, j]] @ v_i_plus
    quantities.W[:, j] = quantities.W[:, [i, j]] @ v_j_plus
    quantities.UQ[:, i] = quantities.UQ[:, [i, j]] @ v_i_plus
    quantities.UQ[:, j] = quantities.UQ[:, [i, j]] @ v_j_plus
    quantities.lambdas[i] = lambda_i_plus
    quantities.lambdas[j] = lambda_j_plus

    print(np.diag(quantities.X @ quantities.get_uput() @ quantities.X.T))
    print(quantities.t)

    return h(quantities, i, j, 0)