"""
features.py

Multivariate feature extraction for WDS time-series sensor data.
Computes first-order temporal differences, rolling window statistics, and
multi-channel hydraulic feature representations while preserving feature alignment.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extracts temporal and statistical features from multivariate sensor streams.

    Attributes:
        include_diffs: Whether to append first-order differences (x_t - x_{t-1}).
        include_rolling: Whether to append rolling mean and rolling std features.
        rolling_window: Window size for rolling statistics.
    """

    def __init__(
        self,
        include_diffs: bool = True,
        include_rolling: bool = False,
        rolling_window: int = 6,
    ):
        self.include_diffs = include_diffs
        self.include_rolling = include_rolling
        self.rolling_window = rolling_window

    def extract_features(
        self,
        df: pd.DataFrame,
        sensor_cols: List[str],
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Extract additional multivariate time-series features from raw sensor columns.

        Args:
            df: DataFrame containing raw sensor columns.
            sensor_cols: List of base sensor feature column names.

        Returns:
            Tuple of:
              - df_features: Expanded DataFrame with original + engineered feature columns.
              - feature_names: List of all feature column names (base + engineered).
        """
        df_out = df.copy()
        feature_names = list(sensor_cols)

        if self.include_diffs:
            diff_cols = []
            for col in sensor_cols:
                diff_col = f"{col}_diff"
                df_out[diff_col] = df_out[col].diff().fillna(0.0)
                diff_cols.append(diff_col)
            feature_names.extend(diff_cols)

        if self.include_rolling:
            roll_mean_cols = []
            roll_std_cols = []
            for col in sensor_cols:
                rm_col = f"{col}_roll_mean_{self.rolling_window}"
                rs_col = f"{col}_roll_std_{self.rolling_window}"
                df_out[rm_col] = (
                    df_out[col]
                    .rolling(window=self.rolling_window, min_periods=1)
                    .mean()
                    .fillna(0.0)
                )
                df_out[rs_col] = (
                    df_out[col]
                    .rolling(window=self.rolling_window, min_periods=1)
                    .std()
                    .fillna(0.0)
                )
                roll_mean_cols.append(rm_col)
                roll_std_cols.append(rs_col)
            feature_names.extend(roll_mean_cols)
            feature_names.extend(roll_std_cols)

        logger.info(
            "FeatureExtractor: Expanded %d base features to %d total features",
            len(sensor_cols),
            len(feature_names),
        )
        return df_out, feature_names
