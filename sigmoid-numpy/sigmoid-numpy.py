import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    Works for scalars, lists, and NumPy arrays.
    Returns a NumPy array (or NumPy scalar) of floats.
    """
    x = np.asarray(x, dtype=float)  # Hint 2: ensure vectorized operations
    return 1.0 / (1.0 + np.exp(-x))