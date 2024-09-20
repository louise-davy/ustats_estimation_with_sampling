import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from metric_learning.build_batches import BuildBatches
from metric_learning.utils import hinge
from metric_learning.sgd import positive_cone, loss

# Config
N_ITER = 2
B = 2
LR = 0.01
batch_sizes_no_pair = np.arange(3, 20, 1)
batch_sizes_pairs = [
    int((batch_size * (batch_size - 1)) / 2) for batch_size in batch_sizes_no_pair
]
equal_probabilities = [True, False]
replacements = [True, False]

# Load data
path_to_data = os.path.join("data", "data_50000_10000_10000.npz")
npz_data = np.load(path_to_data)
X_train = npz_data["X_train"]
y_train = npz_data["y_train"]
X_val = npz_data["X_val"]
y_val = npz_data["y_val"]
X_test = npz_data["X_test"]
y_test = npz_data["y_test"]

# Equal weights at the beginning
W_train = np.ones(X_train.shape[0])
W_val = np.ones(X_val.shape[0])
W_test = np.ones(X_test.shape[0])

# Initialize metric
# M = np.eye(X_train.shape[1])


# equal prob and replacement = Monte-carlo
# equal prob and no replacement = sgd
# unequal prob and replacement = not implemented
# unequal prob and no replacement = poisson


# erm
def erm(batches, sample_pair, M, LR, X_val, y_val, B):
    losses_for_this_config = []
    for batch in batches:
        # perform training
        X_batch = [b[0] for b in batch]
        y_batch = [b[1] for b in batch]

        if sample_pair:
            Wij_batch = [b[2] for b in batch]
        else:
            Wi_batch = [b[2] for b in batch]
            Wj_batch = [b[3] for b in batch]

        X_batch = np.array(X_batch)
        y_batch = np.array(y_batch)
        all_dists = np.array(X_batch @ M @ X_batch.T)  # (batch_size, batch_size)
        hinge_mask = hinge(all_dists, y_batch, B) > 0  # (batch_size, batch_size)
        hinge_mask_sum = np.sum(hinge_mask, axis=1)  # (batch_size,)
        gradient = np.sum(
            (y_batch[:, None] * hinge_mask_sum[:, None]) * X_batch,  # (batch_size, d)
            axis=0,
        )

        M = positive_cone(M - LR * gradient)

        all_dists_val = np.array(X_val @ M @ X_val.T)
        hinge_losses = hinge(all_dists_val, y_val, B)
        total_loss = np.mean(hinge_losses)

        losses_for_this_config.append(total_loss)

    return losses_for_this_config


def save_loss(losses_for_this_config, replacement, equal_prob, all_means, all_stds):
    if replacement and equal_prob:
        all_means["monte_carlo"].append(np.mean(losses_for_this_config))
        all_stds["monte_carlo"].append(np.std(losses_for_this_config))
    elif not replacement and equal_prob:
        all_means["sgd"].append(np.mean(losses_for_this_config))
        all_stds["sgd"].append(np.std(losses_for_this_config))
    elif not equal_prob and not replacement:
        all_means["poisson"].append(np.mean(losses_for_this_config))
        all_stds["poisson"].append(np.std(losses_for_this_config))
    else:
        raise NotImplementedError("This sampling type is not implemented yet.")

    return all_means, all_stds


def run_experiment(
    replacement,
    equal_prob,
    pairs,
    batch_sizes,
    X_train,
    y_train,
    W_train,
    N_ITER,
    X_val,
    y_val,
    LR,
    B,
    all_mean_losses,
    all_std_losses,
):

    print(
        f"Starting config: remise = {replacement}, equal_prob={equal_prob}, pairs = {pairs}"
    )
    means_config = []
    stds_config = []

    for batch_size in batch_sizes:
        M = np.random.randn(X_train.shape[1], X_train.shape[1])
        # Handle batch size computation based on whether we're using pairs or not
        if pairs:
            print(f"Batch size: {batch_size}")
        else:
            print(f"Batch size: {int(batch_size * (batch_size - 1) / 2)}")

        # Build batches
        build = BuildBatches(X_train, y_train, W_train, batch_size, N_ITER)
        batches = build.build_batches(
            replacement=replacement, equal_probabilties=equal_probabilities, pairs=pairs
        )

        # Compute losses using erm function
        losses_for_this_config = erm(batches, pairs, M, LR, X_val, y_val, B)

        all_mean_losses, all_std_losses = save_loss(
            losses_for_this_config,
            replacement,
            equal_prob,
            all_mean_losses,
            all_std_losses,
        )

        means = np.mean(losses_for_this_config)
        stds = np.std(losses_for_this_config)
        mins = np.min(losses_for_this_config)
        maxs = np.max(losses_for_this_config)

        means_config.append(means)
        stds_config.append(stds)

        # Temporary save
        if pairs:
            with open("results/temp_means_pairs.pkl", "wb") as f:
                pickle.dump(all_mean_losses, f)
            with open("results/temp_stds_pairs.pkl", "wb") as f:
                pickle.dump(all_std_losses, f)
        else:
            with open("results/temp_means_observation.pkl", "wb") as f:
                pickle.dump(all_mean_losses, f)
            with open("results/temp_stds_observation.pkl", "wb") as f:
                pickle.dump(all_std_losses, f)

        # Print statistics
        print(
            f"Mean loss: {means:.4f} | Min loss: {mins:.4f} | Max loss: {maxs:.4f} | Std loss: {stds:.4f}"
        )

    return means_config, stds_config, all_mean_losses, all_std_losses


# Main loop for both individual observations and pairs
# Build batches per observation
all_mean_losses_observation = {"monte_carlo": [], "poisson": [], "sgd": []}
all_std_losses_observation = {"monte_carlo": [], "poisson": [], "sgd": []}

# Build batches per pair
all_mean_losses_pairs = {"monte_carlo": [], "poisson": [], "sgd": []}
all_std_losses_pairs = {"monte_carlo": [], "poisson": [], "sgd": []}

for replacement in replacements:
    for equal_prob in equal_probabilities:
        if not equal_prob and replacement:
            continue

        # Individual observations case
        (
            means_config,
            stds_config,
            all_mean_losses_observation,
            all_std_losses_observation,
        ) = run_experiment(
            replacement=replacement,
            equal_prob=equal_prob,
            pairs=False,  # Change to True for pairs
            batch_sizes=batch_sizes_no_pair,
            X_train=X_train,
            y_train=y_train,
            W_train=W_train,
            N_ITER=N_ITER,
            X_val=X_val,
            y_val=y_val,
            LR=LR,
            B=B,
            all_mean_losses=all_mean_losses_observation,
            all_std_losses=all_std_losses_observation,
        )

        # Plot for individual observations
        plt.errorbar(
            [
                int(batch)
                for batch in batch_sizes_no_pair * (batch_sizes_no_pair - 1) / 2
            ],
            means_config,
            yerr=stds_config,
            label=f"repl: {replacement}, eq: {equal_prob}, pairs=False",
        )
        plt.xlabel("Batch size")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

        # Pairs case
        means_config, stds_config, all_mean_losses_pairs, all_std_losses_pairs = (
            run_experiment(
                replacement=replacement,
                equal_prob=equal_prob,
                pairs=True,
                batch_sizes=batch_sizes_pairs,
                X_train=X_train,
                y_train=y_train,
                W_train=W_train,
                N_ITER=N_ITER,
                X_val=X_val,
                y_val=y_val,
                LR=LR,
                B=B,
                all_mean_losses=all_mean_losses_pairs,
                all_std_losses=all_std_losses_pairs,
            )
        )

        # Plot for pairs
        plt.errorbar(
            batch_sizes_pairs,
            means_config,
            yerr=stds_config,
            label=f"repl: {replacement}, eq: {equal_prob}, pairs=True",
        )
        plt.xlabel("Batch size (pairs)")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()


# Save results
import pickle

with open("results/all_mean_losses_observation.pkl", "wb") as f:
    pickle.dump(all_mean_losses_observation, f)

with open("results/all_std_losses_observation.pkl", "wb") as f:
    pickle.dump(all_std_losses_observation, f)

with open("results/all_mean_losses_pairs.pkl", "wb") as f:
    pickle.dump(all_mean_losses_pairs, f)

with open("results/all_std_losses_pairs.pkl", "wb") as f:
    pickle.dump(all_std_losses_pairs, f)
