import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class Normalizer(ABC):
    """
    Dataset normalization.
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def fit(
        self, 
        df: pd.DataFrame, 
        mask: pd.DataFrame
    ) -> "Normalizer":
        """
        Compute normalization statistics from **observed** entries only,
        so that missingness does not bias the normalization.

        Args:
            - df (pd.DataFrame): Raw data where NaN = originally missing.
            - mask (pd.DataFrame): Boolean dataframe where True = observed, False = missing.
        """

    @abstractmethod
    def transform(
        self, 
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """Apply the fitted normalization to df."""

    @abstractmethod
    def inverse_transform(
        self, 
        values: np.ndarray
    ) -> np.ndarray:
        """
        Reverse the normalization.

        Args:
            - values (np.ndarray): Values array in normalized space.

        Returns:
            - (np.ndarray): Array in the original data space, same shape as input.
        """


class StandardNormalizer(Normalizer):
    """
    Per-feature z-score normalization.

    Statistics are fitted only on observed (non-NaN) entries so that
    missingness does not bias the mean or standard deviation.
    Features with zero variance are left with std=1 to avoid division by zero.
    """

    def __init__(self) -> "StandardNormalizer":
        self.means: np.ndarray | None = None
        self.stds: np.ndarray | None = None

    @property
    def name(self) -> str:
        return "standard"

    def fit(
        self, 
        df: pd.DataFrame, 
        mask: pd.DataFrame
    ) -> "StandardNormalizer":
        x = df.values.astype(float)
        m = mask.values.astype(bool)
        observed = np.where(m, x, np.nan)
        self.means = np.nanmean(observed, axis=0)
        self.stds = np.nanstd(observed, axis=0)
        self.stds[self.stds == 0] = 1.0
        logger.debug(
            f"\n Z-Score normalization:"
            f"\n Mean (observed only): {self.means}" # should be around 0
            f"\n Std (observed only): {self.stds}" # should be around 1
        )
        return self

    def transform(
        self, 
        df: pd.DataFrame
    ) -> pd.DataFrame:
        self._check_fitted()
        scaled = (df.values.astype(float) - self.means) / self.stds
        logger.debug(
            f"\n df: {df}"
            f"\n Z-Score normalization:"
            f"\n Min: {df.values.min}"
            f"\n Max: {df.values.max}"
        )
        return pd.DataFrame(scaled, index=df.index, columns=df.columns)
    
    def fit_transform(
        self,
        df: pd.DataFrame, 
        mask: pd.DataFrame
    ) -> pd.DataFrame:
        scaler = self.fit(df, mask)
        return scaler.transform(df)

    def inverse_transform(
        self, 
        values: np.ndarray
    ) -> np.ndarray:
        self._check_fitted()
        return values * self.stds + self.means

    def _check_fitted(self) -> None:
        if self.means is None or self.stds is None:
            raise RuntimeError("Call fit() before.")


class MinMaxNormalizer(Normalizer):
    """
    Per-feature min-max normalization into [0, 1].

    Statistics are fitted only on observed (non-NAN) entries.
    Features where min == max are mapped to 0 to avoid division by zero.
    """

    def __init__(self) -> None:
        self.mins: np.ndarray | None = None
        self.maxs: np.ndarray | None = None
        self.ranges: np.ndarray | None = None

    @property
    def name(self) -> str:
        return "minmax"

    def fit(
        self, 
        df: pd.DataFrame, 
        mask: pd.DataFrame
    ) -> "MinMaxNormalizer":
        x = df.values.astype(float)
        m = mask.values.astype(bool)
        observed = np.where(m, x, np.nan)
        self.mins = np.nanmin(observed, axis=0)
        self.maxs = np.nanmax(observed, axis=0)
        self.ranges = self.maxs - self.mins
        self.ranges[self.ranges == 0] = 1.0
        return self

    def transform(
        self, 
        df: pd.DataFrame
    ) -> pd.DataFrame:
        self._check_fitted()
        scaled = (df.values.astype(float) - self.mins) / self.ranges
        return pd.DataFrame(scaled, index=df.index, columns=df.columns)
    
    def fit_transform(
        self,
        df: pd.DataFrame, 
        mask: pd.DataFrame
    ) -> pd.DataFrame:
        scaler = self.fit(df, mask)
        return scaler.transform(df)

    def inverse_transform(
        self, 
        values: np.ndarray
    ) -> np.ndarray:
        self._check_fitted()
        return values * self.ranges + self.mins

    def _check_fitted(self) -> None:
        if self.mins is None or self.maxs is None or self.ranges is None:
            raise RuntimeError("Call fit() before.")