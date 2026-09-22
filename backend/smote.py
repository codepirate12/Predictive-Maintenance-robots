"""
Minimal SMOTE (Synthetic Minority Over-sampling Technique) implementation.

Built from scratch with scikit-learn's NearestNeighbors so the project does
not need the external `imbalanced-learn` package as a dependency (it is not
installed in every environment and pulling in a new dependency for one
function is unnecessary). This implements the same core algorithm
(Chawla et al., 2002): for each minority-class sample, find its k nearest
minority-class neighbors and generate synthetic points along the line
segments joining them.

Only intended to be used on the TRAINING split. Never apply this to a
validation/test split -- that would leak synthetic near-duplicates across
the split and inflate evaluation metrics.
"""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def smote_oversample(
    X: np.ndarray,
    y: np.ndarray,
    minority_label: int = 1,
    k_neighbors: int = 5,
    target_ratio: float = 1.0,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Oversample the minority class in (X, y) using SMOTE.

    Args:
        X: feature matrix, shape (n_samples, n_features). Must be purely
           numeric (one-hot encode categoricals before calling this).
        y: binary label vector, shape (n_samples,).
        minority_label: the label value to oversample.
        k_neighbors: number of nearest minority neighbors considered when
           synthesizing new points.
        target_ratio: desired ratio of minority-count / majority-count
           after oversampling. 1.0 = fully balanced classes.
        random_state: seed for reproducibility.

    Returns:
        (X_resampled, y_resampled) with synthetic minority samples appended.
    """
    rng = np.random.RandomState(random_state)

    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    minority_mask = y == minority_label
    X_min = X[minority_mask]
    n_minority = len(X_min)
    n_majority = len(y) - n_minority

    if n_minority == 0:
        raise ValueError("No samples found for the given minority_label.")

    n_needed = int(round(target_ratio * n_majority)) - n_minority
    if n_needed <= 0:
        # Already balanced enough -- nothing to do.
        return X, y

    k = min(k_neighbors, n_minority - 1)
    if k < 1:
        # Not enough minority samples to find neighbors; duplicate instead.
        idx = rng.randint(0, n_minority, size=n_needed)
        synthetic = X_min[idx]
    else:
        nn = NearestNeighbors(n_neighbors=k + 1).fit(X_min)
        _, neighbor_idx = nn.kneighbors(X_min)
        # Drop the first column: a point's own nearest "neighbor" is itself.
        neighbor_idx = neighbor_idx[:, 1:]

        synthetic = np.empty((n_needed, X.shape[1]), dtype=float)
        for i in range(n_needed):
            sample_i = rng.randint(0, n_minority)
            neighbor_i = neighbor_idx[sample_i, rng.randint(0, k)]
            gap = rng.rand()
            synthetic[i] = X_min[sample_i] + gap * (X_min[neighbor_i] - X_min[sample_i])

    X_resampled = np.vstack([X, synthetic])
    y_resampled = np.concatenate([y, np.full(n_needed, minority_label)])

    # Shuffle so synthetic rows aren't all appended at the end.
    shuffle_idx = rng.permutation(len(y_resampled))
    return X_resampled[shuffle_idx], y_resampled[shuffle_idx]
