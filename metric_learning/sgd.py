import numpy as np
from time import time
from metric_learning.utils import (
    create_fast,
    hinge,
    loss,
    create_mini_batch,
)
import itertools


def run_sgd(
    X_train: np.array,
    y_train: np.array,
    X_val: np.array,
    y_val: np.array,
    n_iter: int,
    batch_size: int,
    b: int,
    M: np.array,
    lr: float,
    sampling_type: str,
    sample_pairs: bool,
    times: list,
    print_every: int = 500,
    W=None,
):
    start = time()
    if sampling_type == "inequal_without_replacement":
        if W is None:
            # equal weights for first step
            W = np.ones(X_train.shape[0])
    history = {}
    losses = []
    losses.append(loss(M, X_val, y_val, b=b))
    Ms = []
    Ms.append(M)
    if not sample_pairs:
        for i in range(n_iter):
            if sampling_type == "equal_without_replacement":
                all_indices = np.arange(X_train.shape[0])
                batch_indices = all_indices[i * batch_size : (i + 1) * batch_size]
                X_batch = X_train[batch_indices]
                y_batch = y_train[batch_indices]
                W_batch = W[batch_indices]
            else:
                X_batch, y_batch, W_batch, indices = create_mini_batch(
                    X_train, y_train, batch_size, sampling_type=sampling_type, W=W
                )
            M, losses = sgd_on_batch(
                X_batch, y_batch, X_val, y_val, M, W_batch, b, lr, losses
            )
            if i % print_every == 0:
                print(f"iter {i+1}, score {losses[i+1]}")

                if i > 0:
                    if losses[i + 1] >= losses[i - print_every]:
                        break
    else:
        pairs = itertools.combinations(range(X_train.shape[0]), 2)

    history["losses"] = losses
    history["Ms"] = Ms
    times.append(time() - start)

    return M, history, times


def grad_dist_without_pairs(M, X, y, b=2, fast_X=None):
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


def grad_dist_with_pairs(M, X, y, b=2, fast_X=None, W=None):
    """Gradient of the hinge loss with pairs."""
    if fast_X is None:
        fast_X = create_fast(X)
    pair_y = np.kron(y, y)
    all_dists = fast_X @ M.ravel()
    return np.sum(
        np.outer(pair_y * (hinge(all_dists, pair_y, b) > 0), np.ones(fast_X.shape[1]))
        * fast_X
        * W[:, np.newaxis],
        axis=0,
    ).reshape(M.shape)


def positive_cone(M):
    """Project on the semi-definite positive cone."""
    eigval, eigvec = np.linalg.eigh(M)
    projected_eigval = (eigval > 0) * eigval
    return (projected_eigval * eigvec.T) @ eigvec


def simple_link_function(Wi, Wj, M):
    return Wi @ M @ Wj.T


def logistic_link_function(Wi, Wj, M):
    return 1 / (1 + np.exp(-Wi @ M @ Wj.T))


def sgd_on_batch(
    X_batch, y_batch, X_val, y_val, M, W_batch, b, lr, losses, sampling_pairs
):
    fast_X_batch = create_fast(X_batch)
    if not sampling_pairs:
        gradient = grad_dist_without_pairs(
            M, X_batch, y_batch, b, fast_X_batch, W_batch
        )
    else:
        gradient = grad_dist_with_pairs(M, X_batch, y_batch, b, fast_X_batch, W_batch)
    M = positive_cone(M - lr * gradient)

    # score on val
    losses.append(loss(M, X_val, y_val, b=b))

    return M, losses
