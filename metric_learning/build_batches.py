import itertools
import numpy as np


class BuildBatches:
    def __init__(self, X, y, W, batch_size, n_iter):
        self.X = X
        self.y = y
        self.W = W
        self.batch_size = batch_size
        self.n_iter = n_iter

    def build_batches(self, replacement: bool, equal_probabilties: bool, pairs: bool):
        if replacement:
            if equal_probabilties:
                if pairs:
                    return self.batches_pairs_with_replacement()
                else:
                    return self.batches_observations_with_replacement()
            else:
                if pairs:
                    return self.batches_pairs_unequal_probabilities()
                else:
                    return self.batches_observations_unequal_probabilities()
        else:
            if equal_probabilties:
                if pairs:
                    return self.batches_pairs_without_replacement()
                else:
                    return self.batches_observations_without_replacement()
            else:
                if pairs:
                    return self.batches_pairs_unequal_probabilities()
                else:
                    return self.batches_observations_unequal_probabilities()

    def batches_observations_with_replacement(self):
        batches = []
        for i in range(self.n_iter):
            pairs = itertools.combinations(range(self.batch_size), 2)
            indices = np.random.choice(self.X.shape[0], self.batch_size, replace=True)
            X_batch = self.X[indices]
            y_batch = self.y[indices]
            one_batch = []
            for pair in pairs:
                y_diff = int(y_batch[pair[0]] == y_batch[pair[1]])
                x_diff = X_batch[pair[0]] - X_batch[pair[1]]
                one_batch.append((x_diff, y_diff, 1, 1))
            batches.append(one_batch)
        return batches

    def batches_observations_without_replacement(self):
        # shuffle dataset
        indices = np.random.permutation(self.X.shape[0])
        X = self.X[indices]
        y = self.y[indices]
        batches = []
        for i in range(self.n_iter):
            pairs = itertools.combinations(range(self.batch_size), 2)
            X_batch = X[i * self.batch_size : (i + 1) * self.batch_size]
            y_batch = y[i * self.batch_size : (i + 1) * self.batch_size]
            one_batch = []
            for pair in pairs:
                y_diff = int(y_batch[pair[0]] == y_batch[pair[1]])
                x_diff = X_batch[pair[0]] - X_batch[pair[1]]
                one_batch.append((x_diff, y_diff, 1, 1))
            batches.append(one_batch)
        return batches

    def batches_pairs_with_replacement(self):
        list_indices = self.X.shape[0]
        batches = []
        for i in range(self.n_iter):
            indice_i = np.random.choice(list_indices, self.batch_size, replace=True)
            indice_j = np.random.choice(list_indices, self.batch_size, replace=True)
            pairs = zip(indice_i, indice_j)
            one_batch = []
            for pair in pairs:
                y_diff = int(self.y[pair[0]] == self.y[pair[1]])
                x_diff = self.X[pair[0]] - self.X[pair[1]]
                one_batch.append((x_diff, y_diff, 1, 1))
            batches.append(one_batch)
        return batches

    def batches_pairs_without_replacement(self):
        batches = []
        for i in range(self.n_iter):
            indices_i = np.random.permutation(self.X.shape[0])
            indices_j = np.random.permutation(self.X.shape[0])

            mask = indices_i != indices_j
            indices_i = indices_i[mask]
            indices_j = indices_j[mask]
            if len(indices_i) < self.batch_size or len(indices_j) < self.batch_size:
                raise ValueError("Not enough unique pairs to form a batch")

            indices_i = indices_i[: self.batch_size]
            indices_j = indices_j[: self.batch_size]

            pairs = zip(indices_i, indices_j)
            one_batch = []
            for pair in pairs:
                y_diff = int(self.y[pair[0]] == self.y[pair[1]])
                x_diff = self.X[pair[0]] - self.X[pair[1]]
                one_batch.append((x_diff, y_diff, 1, 1))
            batches.append(one_batch)
        return batches

    def batches_observations_unequal_probabilities(self):
        # shuffle dataset
        indices = np.random.permutation(self.X.shape[0])
        X = self.X[indices]
        y = self.y[indices]
        W = self.W[indices]
        batches = []
        for i in range(self.n_iter):
            pairs = itertools.combinations(range(self.batch_size), 2)
            j = i * self.batch_size
            one_batch = []
            new_indices = []
            while len(new_indices) < self.batch_size:
                # tirage bernoulli pour savoir si on tire l'observation ou non
                tirage = np.random.rand()
                if tirage < W[j]:
                    new_indices.append(j)
                else:
                    pass
                j += 1
            X_batch = X[new_indices]
            y_batch = y[new_indices]
            W_batch = W[new_indices]
            for pair in pairs:
                y_diff = int(y_batch[pair[0]] == y_batch[pair[1]])
                x_diff = X_batch[pair[0]] - X_batch[pair[1]]
                W_i = W_batch[pair[0]]
                W_j = W_batch[pair[1]]
                one_batch.append((x_diff, y_diff, W_i, W_j))
            batches.append(one_batch)
        return batches

    def batches_pairs_unequal_probabilities(self):
        indices_i = np.random.permutation(self.X.shape[0])
        indices_j = np.random.permutation(self.X.shape[0])
        mask = indices_i != indices_j
        indices_i = indices_i[mask]
        indices_j = indices_j[mask]
        batches = []
        for i in range(self.n_iter):
            new_indices = []
            j = i * self.batch_size

            # Ensure enough valid pairs
            while len(new_indices) < self.batch_size and j < len(indices_i):
                # Bernoulli sampling for pairs
                tirage = np.random.rand()
                if tirage < self.W[j]:
                    new_indices.append(j)
                j += 1

            if len(new_indices) < self.batch_size:
                raise ValueError("Not enough valid pairs for the batch.")

            indices_i_batch = [indices_i[k] for k in new_indices]
            indices_j_batch = [indices_j[k] for k in new_indices]

            pairs = zip(indices_i_batch, indices_j_batch)
            one_batch = []
            for pair in pairs:
                y_diff = int(self.y[pair[0]] == self.y[pair[1]])
                x_diff = self.X[pair[0]] - self.X[pair[1]]
                W_ij = self.W[pair[0], pair[1]]
                one_batch.append((x_diff, y_diff, W_ij))
            batches.append(one_batch)
        return batches
