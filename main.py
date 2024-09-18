import numpy as np
from metric_learning.utils import (
    plot_results,
)
from metric_learning.sgd import run_sgd
import os
import yaml
import argparse
import sys

# Parse arguments
parser = argparse.ArgumentParser(
    prog="Prepare data for metric learning",
    description="It will create the file depending on the sampling type, ratio\
      and n_val provided in the config file",
    epilog="Enjoy the program! :)",
)

parser.add_argument(
    "-c", "--config", type=str, help="Path to the config file for SGD", required=True
)

args = parser.parse_args()

# Read config file
config_file = args.config

# Check if the config file exists
if not os.path.exists(config_file):
    raise ValueError("The config file does not exist")

# Check if the config file is a yaml file
if not config_file.endswith(".yaml"):
    raise ValueError("The config file should be a yaml file")

# Load config file
with open(config_file, "r") as file:
    config = yaml.safe_load(file)

# Get parameters for SGD
STEP = config["LR"]
N_ITER = config["N_ITER"]
N_RUNS = config["N_RUNS"]
B = config["B"]
BATCH_SIZE = config["BATCH_SIZE"]

# Get parameters to loop over
with open("configs/parameters.yaml", "r") as file:
    parameters = yaml.safe_load(file)

print(parameters)
sys.exit()
SAMPLING_TYPE = args.sampling_type
M_INITIALISATION = args.m_initialisation
SAMPLE_PAIRS = args.pairs

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
    n_iter=N_ITER,
    batch_size=BATCH_SIZE,
    b=B,
    M=M,
    lr=STEP,
    sampling_type=SAMPLING_TYPE,
    sample_pairs=SAMPLE_PAIRS,
    times=[],
    print_every=500,
    W=None,
)

# Plot results

plot_results(M, history["losses"])
