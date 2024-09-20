import numpy as np
from metric_learning.utils import (
    plot_results,
    initalise_M,
)
from metric_learning.sgd import run_sgd
import os
import yaml
import argparse

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
DATA_PATH = config["DATA_PATH"]
DATA_FILE = config["DATA_FILE"]

# Get parameters to loop over
with open("configs/parameters.yaml", "r") as file:
    parameters = yaml.safe_load(file)

M_INITIALISATIONS = parameters["M_INITIALISATION"]
SAMPLING_TYPES = parameters["SAMPLING_TYPE"]
SAMPLE_PAIRS = parameters["SAMPLE_PAIRS"]


# Load data
path_to_data = os.path.join(DATA_PATH, DATA_FILE)
npz_data = np.load(path_to_data)
X_train = npz_data["X_train"]
y_train = npz_data["y_train"]
X_val = npz_data["X_val"]
y_val = npz_data["y_val"]
X_test = npz_data["X_test"]
y_test = npz_data["y_test"]

for sample_pair in SAMPLE_PAIRS:
    for sampling_type in SAMPLING_TYPES:
        for m_initialisation in M_INITIALISATIONS:
            # Initialize M
            M = initalise_M(m_initialisation, X_train.shape[1])

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
                sampling_type=sampling_type,
                sample_pairs=sample_pair,
                times=[],
                print_every=500,
                W=None,
            )

            # Plot results

            plot_results(M, history["losses"])
