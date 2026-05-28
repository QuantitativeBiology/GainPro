import numpy as np
from scipy.stats import pearsonr

def compute_metrics(
    true_values: np.ndarray,
    predicted_values: np.ndarray,
) -> tuple[float, float]:
    """
    Compute the Pearson r and RMSE between the true and predicted values.

    Args:
        - true_values (np.ndarray): True values.
        - predicted_values (np.ndarray): Predicted values.

    Returns:
        - tuple(float, float): Return (Pearson r, RMSE) between the true and predicted values.
    """
    r, _ = pearsonr(true_values, predicted_values)
    rmse = np.sqrt(np.mean((true_values - predicted_values) ** 2))
    return r, rmse