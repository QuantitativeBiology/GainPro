import pandas as pd

from config import EntryState

def filter_artificial_missing_entries(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Filter artificially hidden entries.

    Args:
        - predictions (pd.DataFrame): DataFrame containing model predictions.
            Each row represents a single entry with the following fields:
            (sample_id, feature, true_value, predicted_value, observed_mask, artificial_missing_mask, group_id).
    
    Returns:
        - (pd.DataFrame): Subset of `predictions` containing only the artificially hidden entries.
    """
    return predictions[
        (predictions["artificial_missing_mask"] == EntryState.ARTIFICIAL_MISSING) &
        (predictions["observed_mask"] == EntryState.OBSERVED)
    ]

def filter_tissue_entries(
    predictions: pd.DataFrame,
    tissue: str,
) -> pd.DataFrame:
    """
    Filter artificially hidden entries for the given `tissue`.

    Args:
        - predictions (pd.DataFrame): DataFrame containing model predictions.
            Each row represents a single entry with the following fields:
            (sample_id, feature, true_value, predicted_value, observed_mask, artificial_missing_mask, group_id).
        - tissue (str): Tissue to filter.
    
    Returns:
        - (pd.DataFrame): Subset of `predictions` containing only the artificially hidden entries for the given `tissue`.
    """
    return predictions[
        (predictions["group_id"] == tissue) &
        (predictions["artificial_missing_mask"] == EntryState.ARTIFICIAL_MISSING) &
        (predictions["observed_mask"] == EntryState.OBSERVED)
    ]