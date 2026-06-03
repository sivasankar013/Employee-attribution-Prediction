from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def smote_oversample(
    X: np.ndarray,
    y: np.ndarray,
    *,
    random_state: int = 42,
    k_neighbors: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """A lightweight SMOTE-style oversampler without external dependencies."""

    # Make sure we are working with NumPy arrays, not pandas objects.
    X = np.asarray(X)
    y = np.asarray(y)

    # Find the class counts so we know which label is the minority class.
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) != 2:
        # If this is not a binary problem, just return the original data.
        return X, y

    # Identify the minority and majority labels.
    minority_class = classes[np.argmin(counts)]
    majority_class = classes[np.argmax(counts)]
    minority = X[y == minority_class]
    majority = X[y == majority_class]

    # SMOTE needs enough minority samples to build neighbors.
    if len(minority) < 2 or len(majority) <= len(minority):
        return X, y

    # Use a deterministic random generator so results are reproducible.
    rng = np.random.default_rng(random_state)

    # We generate just enough new synthetic rows to balance the classes.
    n_to_generate = len(majority) - len(minority)
    k = min(k_neighbors, len(minority) - 1)
    if k < 1:
        return X, y

    # Fit nearest-neighbor search on the minority class only.
    nn = NearestNeighbors(n_neighbors=k + 1)
    nn.fit(minority)
    neighbor_indices = nn.kneighbors(minority, return_distance=False)

    # Create synthetic samples by interpolating between a point and one of its neighbors.
    synthetic = []
    for _ in range(n_to_generate):
        idx = rng.integers(0, len(minority))
        sample = minority[idx]
        neighbors = neighbor_indices[idx][1:]
        neighbor = minority[rng.choice(neighbors)]
        gap = rng.random()
        synthetic.append(sample + gap * (neighbor - sample))

    # Combine the original data with the new synthetic minority samples.
    X_res = np.vstack([X, np.asarray(synthetic)])
    y_res = np.concatenate([y, np.full(len(synthetic), minority_class)])
    return X_res, y_res
