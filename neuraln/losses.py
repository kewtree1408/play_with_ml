import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Mean Squared Error
    assert y_true.shape == y_pred.shape
    return ((y_true - y_pred)**2).mean()
