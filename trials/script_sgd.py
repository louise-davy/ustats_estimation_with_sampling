#!/usr/bin/env python3

import numpy as np
import scipy
import matplotlib.pyplot as plt


def create_fast(X):
    """Create bigger matrix filled with x - y pairs."""
    n, d = X.shape
    pairs = np.kron(np.ones((n, 1)), X) - np.kron(X, np.ones((n, 1)))
    print("pairs_shape", pairs.shape)
    rk1_matrices = np.kron(np.ones((1, d)), pairs) * np.kron(pairs, np.ones((1, d)))
    print("rk1_matrices_shape", rk1_matrices.shape)
    return rk1_matrices


def hinge(u, y, b):
    """Hinge loss."""
    return np.maximum(0, 1 - y * (b - u))


def create_mahalanobis(M):
    """Squared Mahalanobis distance."""
    return lambda x, y: (x - y) @ M @ (x - y)


def create_fast_mahalanobis(flat_M):
    """Fast version of Mahalanobis (memory-expensive)."""
    return lambda big_x: flat_M @ big_x


def loss(M, X, y, b=2, fast_X=None):
    """Hinge loss for metric learning."""
    if fast_X is None:
        fast_X = create_fast(X)
    # fast_dist = create_fast_mahalanobis(M.ravel())
    all_dists = fast_X @ M.ravel()
    pair_y = np.kron(y, y)
    return np.average(hinge(all_dists, pair_y, b))


def grad_dist(M, X, y, b=2, fast_X=None):
    """Gradient of the hinge loss."""
    if fast_X is None:
        fast_X = create_fast(X)
    pair_y = np.kron(y, y)
    all_dists = fast_X @ M.ravel()
    return np.sum(
        np.outer(pair_y * (hinge(all_dists, pair_y, b) > 0), np.ones(fast_X.shape[1]))
        * fast_X,
        axis=0,
    ).reshape(M.shape)


def positive_cone(M):
    """Project on the semi-definite positive cone."""
    eigval, eigvec = np.linalg.eigh(M)
    projected_eigval = (eigval > 0) * eigval
    return (projected_eigval * eigvec.T) @ eigvec


if __name__ == "__main__":
    n = 100
    d = 2
    mu_0 = np.zeros(d)
    mu_0[0] = 1.0
    mu_1 = np.zeros(d)
    mu_1[0] = -1.0
    cov = 0.01 * np.eye(d)
    X = np.zeros((2 * n, d))
    y = np.zeros(2 * n, dtype=int)
    X[:n] = np.random.multivariate_normal(mu_0, cov, n)
    y[:n] = 1
    X[n:] = np.random.multivariate_normal(mu_1, cov, n)
    y[n:] = -1

    step = 5e-4
    M = np.eye(d)
    n_iter = 500
    b = 1.1
    score = np.zeros(n_iter + 1)
    score[0] = loss(M, X, y, b=b)
    fast_X = create_fast(X)
    for i in range(n_iter):
        M = positive_cone(M - step * grad_dist(M, X, y, b=b, fast_X=fast_X))
        score[i + 1] = loss(M, X, y, b=b, fast_X=fast_X)

    plt.subplot(2, 2, 1)
    plt.plot(score)
    plt.title("Loss")

    plt.subplot(2, 2, 3)
    colors = np.array(["b", "r"])
    plt.scatter(X[:, 0], X[:, 1], color=colors[(y > 0) * y], s=3, alpha=0.5)
    plt.ylim([-0.3, 0.3])
    plt.title("Init")

    plt.subplot(2, 2, 4)
    sqrt_M = scipy.linalg.sqrtm(M)
    transf_X = X @ sqrt_M
    plt.scatter(
        transf_X[:, 0], transf_X[:, 1], color=colors[(y > 0) * y], s=3, alpha=0.5
    )
    # plt.ylim([-0.3, 0.3])
    plt.title("Final")

    plt.subplot(2, 2, 2)
    plt.imshow(M)
    plt.tight_layout()
    plt.show()
