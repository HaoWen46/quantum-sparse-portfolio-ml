"""Feature engineering helpers for Phase 2 forecasting."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from sparse_portfolio.data import returns_from_prices


def make_return_features(
    prices: pd.DataFrame,
    *,
    return_windows: Iterable[int] = (1, 5, 20),
    volatility_windows: Iterable[int] = (20,),
) -> pd.DataFrame:
    """Create a long date/ticker feature frame using only past information."""

    prices = prices.sort_index()
    returns = returns_from_prices(prices)
    frames: list[pd.Series] = []

    for window in return_windows:
        if window <= 0:
            raise ValueError("return windows must be positive")
        stacked = prices.pct_change(window, fill_method=None).stack(future_stack=True)
        frames.append(stacked.rename(f"ret_{window}d"))

    for window in volatility_windows:
        if window <= 1:
            raise ValueError("volatility windows must be greater than 1")
        stacked = returns.rolling(window).std().stack(future_stack=True)
        frames.append(stacked.rename(f"vol_{window}d"))

    features = pd.concat(frames, axis=1)
    features.index.names = ["date", "ticker"]
    return features.dropna()


def make_forward_returns(prices: pd.DataFrame, *, target_horizon: int = 5) -> pd.Series:
    """Create long forward-return targets."""

    if target_horizon <= 0:
        raise ValueError("target_horizon must be positive")

    forward_return = (
        prices.sort_index()
        .shift(-target_horizon)
        .divide(prices)
        .subtract(1.0)
        .stack(future_stack=True)
        .rename(f"fwd_ret_{target_horizon}d")
    )
    forward_return.index.names = ["date", "ticker"]
    return forward_return.dropna()


def make_return_feature_panel(
    prices: pd.DataFrame,
    *,
    return_windows: Iterable[int] = (1, 5, 20),
    volatility_windows: Iterable[int] = (20,),
    target_horizon: int = 5,
) -> pd.DataFrame:
    """Create a long date/ticker feature panel with forward-return targets."""

    features = make_return_features(
        prices,
        return_windows=return_windows,
        volatility_windows=volatility_windows,
    )
    target = make_forward_returns(prices, target_horizon=target_horizon)
    panel = pd.concat([features, target], axis=1)
    panel.index.names = ["date", "ticker"]
    return panel.dropna()
