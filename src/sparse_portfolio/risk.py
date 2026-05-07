"""Risk-estimation helpers for portfolio optimization."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


def sample_covariance(returns: pd.DataFrame) -> np.ndarray:
    """Return the sample covariance matrix for aligned returns."""

    clean = returns.dropna(axis=0, how="any")
    if len(clean) < 2:
        raise ValueError("need at least two complete return observations")
    return clean.cov().to_numpy()


def ledoit_wolf_covariance(returns: pd.DataFrame) -> np.ndarray:
    """Return a Ledoit-Wolf shrinkage covariance estimate."""

    clean = returns.dropna(axis=0, how="any")
    if len(clean) < 2:
        raise ValueError("need at least two complete return observations")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy")
        covariance = LedoitWolf().fit(clean.to_numpy()).covariance_
    if not np.isfinite(covariance).all():
        raise ValueError("Ledoit-Wolf covariance contains non-finite values")
    if not np.allclose(covariance, covariance.T):
        raise ValueError("Ledoit-Wolf covariance is not symmetric")
    return covariance
