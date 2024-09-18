import numpy as np
from metric_learning.utils import (
    plot_results,
)
from metric_learning.sgd import run_sgd
import yaml

with open("config.yaml", "r") as f:
    config = yaml.load(f)

STEP = config["STEP"]
N_ITER = config["N_ITER"]
B = config["B"]
BATCH_SIZE = config["BATCH_SIZE"]
M_INITIALISATION = config["M_INITIALISATION"]


# Load data
npz_data = np.load("data/mnist.npz")
X_train = npz_data["X_train"]
y_train = npz_data["y_train"]
X_val = npz_data["X_val"]
y_val = npz_data["y_val"]
X_test = npz_data["X_test"]
y_test = npz_data["y_test"]


# Initialize M
if M_INITIALISATION == "identity":
    M = np.eye(X_train.shape[1])
elif M_INITIALISATION == "random":
    M = np.random.randn(X_train.shape[1], X_train.shape[1])
else:
    raise ValueError("M_INITIALISATION should be 'identity' or 'random'.")

# Run SGD
M, history, times = run_sgd(
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    fast_X_val=fast_X_val,
    n_iter=N_ITER,
    batch_size=BATCH_SIZE,
    b=B,
    M=M,
    lr=STEP,
    sampling_type=SAMPLING_TYPE,
    times=[],
)

# Plot results

plot_results(M, history["losses"])
