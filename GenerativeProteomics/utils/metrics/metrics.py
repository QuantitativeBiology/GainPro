import numpy as np

def rmse(
    x_true: np.ndarray,
    x_pred: np.ndarray,
    mask: np.ndarray[bool],
) -> float:
    """Compute RMSE between true and predicted values on masked positions."""
    diff = x_true[mask] - x_pred[mask]
    rmse = np.sqrt(np.mean(diff ** 2))
    return rmse