import numpy as np


def rmse(truth: np.ndarray, predictions: np.ndarray) -> float:
    """Calculates the RMSE between ground truth values and predictions. Accepts single and multiple values.

    Args:
        truth: Truth values. Either of dimension (dim,) or (amount,dim).
        predictions: Predicted values. Either of dimension (dim,) or (amount,dim).

    Returns:
        Root mean squared error.
    """
    return np.sqrt(np.mean((truth - predictions) ** 2))
