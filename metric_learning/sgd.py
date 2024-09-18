import numpy as np
from time import time
from metric_learning.utils import (
    create_fast,
    loss,
    grad_dist,
    positive_cone,
    create_mini_batch,
)


def run_sgd(
    X_train: np.array,
    y_train: np.array,
    X_val: np.array,
    y_val: np.array,
    fast_X_val: np.array,
    n_iter: int,
    batch_size: int,
    b: int,
    M: np.array,
    lr: float,
    sampling_type: str,
    times: list,
    print_every: int = 500,
    W=None,
):
    start = time()
    if sampling_type == "inequal_without_replacement":
        if W is None:
            # equal inclusion probabilities for first step
            W = np.ones(X_train.shape[0]) / X_train.shape[0]
    history = {}
    losses = []
    losses.append(loss(M, X_val, y_val, b=b, fast_X=fast_X_val))
    Ms = []
    Ms.append(M)
    for i in range(n_iter):
        X_batch, y_batch, indices = create_mini_batch(
            X_train, y_train, batch_size, sampling_type=sampling_type, W=W
        )
        fast_X_batch = create_fast(X_batch)
        gradient = grad_dist(M, X_batch, y_batch, b=b, fast_X=fast_X_batch)
        M = positive_cone(M - lr * gradient)
        if sampling_type == "inequal_without_replacement":
            gradient_norms = np.sum(np.linalg.norm(gradient, axis=1))
            W[indices] = W[indices] + 0.01 * gradient_norms
            W /= np.sum(W)

            # check if nan
            if np.isnan(W).any():
                print(list(W))
                break

        # score on val
        losses.append(loss(M, X_val, y_val, b=b, fast_X=fast_X_val))

        if i % print_every == 0:
            print(f"iter {i+1}, score {losses[i+1]}")

            if i > 0:
                if losses[i + 1] >= losses[i - print_every]:
                    break

    history["losses"] = losses
    history["Ms"] = Ms
    times.append(time() - start)

    return M, history, times
