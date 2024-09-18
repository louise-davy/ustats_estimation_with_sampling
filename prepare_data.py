import numpy as np
from metric_learning.utils import (
    get_data,
    get_validation_dataset,
    transform_data,
)
from datetime import datetime
import yaml
import argparse
import os

# Parse arguments
parser = argparse.ArgumentParser(
    prog="Prepare data for metric learning",
    description="It will create the file depending on the sampling type, ratio\
      and n_val provided in the config file",
    epilog="Enjoy the program! :)",
)

parser.add_argument(
    "-c", "--config", type=str, help="Path to the config file", required=True
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

# Get parameters for data preparation
RATIO_VAL = float(config["RATIO_VAL"])
N_VAL = int(config["N_VAL"])

# Get data
X_train, y_train, X_test, y_test = get_data()

# Split dataset
X_train, y_train, X_val, y_val = get_validation_dataset(
    X_train, y_train, ratio=RATIO_VAL, n_val=N_VAL
)

# Transform data
X_train, pca = transform_data(X_train, fit=True, pca=None)
X_val, _ = transform_data(X_val, fit=False, pca=pca)
X_test, _ = transform_data(X_test, fit=False, pca=pca)

# fast_X_val = create_fast(X_val)
date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


dict_data = {
    "X_train": X_train,
    "y_train": y_train,
    "X_val": X_val,
    "y_val": y_val,
    "X_test": X_test,
    "y_test": y_test,
    "time_creation": np.array(date),  # just in case
}

# Save data
n_train = X_train.shape[0]
n_val = X_val.shape[0]
n_test = X_test.shape[0]
np.savez(f"data/data_{n_train}_{n_val}_{n_test}.npz", **dict_data)
