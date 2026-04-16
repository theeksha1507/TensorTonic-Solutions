import numpy as np

def _entropy(y):
    y = np.asarray(y)
    if y.size == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())

def information_gain(y, split_mask):
    y = np.asarray(y)
    split_mask = np.asarray(split_mask, dtype=bool)

    n = y.size
    if n == 0:
        return 0.0

    y_left = y[split_mask]
    y_right = y[~split_mask]

    n_left = y_left.size
    n_right = y_right.size

    if n_left == 0 or n_right == 0:
        return 0.0

    H_parent = _entropy(y)
    H_left = _entropy(y_left)
    H_right = _entropy(y_right)

    H_children = (n_left / n) * H_left + (n_right / n) * H_right

    return float(H_parent - H_children)