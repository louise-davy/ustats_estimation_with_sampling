import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize


def create_fast(X):
    """Create bigger matrix filled with x - y pairs."""
    n, d = X.shape
    pairs = np.kron(np.ones((n, 1)), X) - np.kron(X, np.ones((n, 1)))
    rk1_matrices = np.kron(np.ones((1, d)), pairs) * np.kron(pairs, np.ones((1, d)))
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


def create_mini_batch(
    X, y, batch_size, sampling_type="equal_without_replacement", W=None
):
    """Randomly select a mini-batch from the data."""
    if sampling_type == "equal_without_replacement":
        raise NotImplementedError("This sampling type is not implemented yet.")
        # indices = np.random.choice(X.shape[0], batch_size, replace=False)
    elif sampling_type == "equal_with_replacement":  # monte carlo
        indices = np.random.choice(X.shape[0], batch_size, replace=True)
    elif sampling_type == "inequal_without_replacement":  # poisson
        indices = np.random.choice(X.shape[0], batch_size, replace=False, p=W)

    return X[indices], y[indices], W[indices], indices


def get_data():
    X_train = np.load("data/mnist_train.npy")
    X_test = np.load("data/mnist_test.npy")
    y_train = np.load("data/mnist_train_labels.npy")
    y_test = np.load("data/mnist_test_labels.npy")
    return X_train, y_train, X_test, y_test


def get_validation_dataset(X, y, ratio, n_val):
    shuffled_indices = np.random.permutation(X.shape[0])
    X, y = X[shuffled_indices], y[shuffled_indices]

    if n_val == 0:
        if ratio == 0:
            raise ValueError("Either ratio or n_val should be provided.")
        else:
            n_val = int(ratio * X.shape[0])
    else:
        if ratio != 0:
            raise ValueError("Either ratio or n_val should be provided.")
        else:
            pass

    n_train = X.shape[0] - n_val

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:], y[n_train:]

    print(f"Train: {X_train.shape}, Val: {X_val.shape}")

    return X_train, y_train, X_val, y_val


def transform_data(X, fit=True, pca=None):
    if fit:
        pca = PCA(n_components=164, svd_solver="full")
        pca.fit(X)
    else:
        if pca is None:
            raise ValueError("pca should be provided if fit is False.")
        else:
            pass
    X = pca.transform(X)
    X = normalize(X, norm="l2", axis=1)

    return X, pca


def plot_results(M, losses):
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title("Loss")

    # plt.subplot(2, 2, 3)
    # colors = np.array(["b", "r"])
    # plt.scatter(X_val[:, 0], X_val[:, 1], c=y_val, s=3, alpha=0.5)
    # plt.ylim([-0.3, 0.3])
    # plt.title("Init")

    # plt.subplot(2, 2, 4)
    # sqrt_M = scipy.linalg.sqrtm(M)
    # transf_X = X_val @ sqrt_M
    # plt.scatter(transf_X[:, 0], transf_X[:, 1], c=y_val, s=3, alpha=0.5)
    # # plt.ylim([-0.3, 0.3])
    # plt.title("Final")

    plt.subplot(1, 2, 2)
    plt.imshow(M)
    plt.colorbar()
    plt.show()


def initialise_M(m_initialisation, d):
    if m_initialisation == "identity":
        M = np.eye(d)
    elif m_initialisation == "random":
        M = np.random.randn(d, d)
    else:
        raise ValueError("m_initialisation should be 'identity' or 'random'.")
    return M
