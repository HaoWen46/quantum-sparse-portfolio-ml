"""Forecasting input helpers for sparse portfolio optimization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, LassoCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sparse_portfolio.data import returns_from_prices
from sparse_portfolio.features import make_forward_returns, make_return_features
from sparse_portfolio.risk import ledoit_wolf_covariance, sample_covariance


@dataclass(frozen=True)
class ForecastInputs:
    """Aligned inputs for one portfolio-selection date."""

    tickers: list[str]
    mu: np.ndarray
    sigma: np.ndarray
    returns_window: pd.DataFrame


def historical_mean_forecast(
    returns: pd.DataFrame,
    *,
    lookback: int | None = None,
) -> pd.Series:
    """Forecast expected returns with a trailing historical mean."""

    window = returns.tail(lookback) if lookback is not None else returns
    clean = window.dropna(axis=0, how="any")
    if clean.empty:
        raise ValueError("no complete return observations for historical mean")
    return clean.mean()


def latest_forecast_inputs(
    prices: pd.DataFrame,
    *,
    lookback: int = 252,
    covariance: str = "ledoit_wolf",
) -> ForecastInputs:
    """Create latest `mu` and `Sigma` inputs from cached price data."""

    if lookback < 2:
        raise ValueError("lookback must be at least 2")

    returns = returns_from_prices(prices)
    window = returns.tail(lookback).dropna(axis=1, how="any").dropna(axis=0, how="any")
    if window.shape[0] < 2:
        raise ValueError("not enough complete returns in lookback window")
    if window.shape[1] == 0:
        raise ValueError("no tickers have complete returns in lookback window")

    mu = historical_mean_forecast(window).to_numpy()
    if covariance == "ledoit_wolf":
        sigma = ledoit_wolf_covariance(window)
    elif covariance == "sample":
        sigma = sample_covariance(window)
    else:
        raise ValueError("covariance must be 'ledoit_wolf' or 'sample'")

    return ForecastInputs(
        tickers=list(window.columns),
        mu=mu,
        sigma=sigma,
        returns_window=window,
    )


def latest_ml_return_forecast(
    prices: pd.DataFrame,
    *,
    model: str = "lasso",
    target_horizon: int = 5,
    train_lookback_rows: int = 2_000,
    random_state: int = 0,
) -> pd.Series:
    """Forecast latest forward returns with a simple course-covered ML model."""

    features = make_return_features(prices)
    targets = make_forward_returns(prices, target_horizon=target_horizon)
    training = features.join(targets, how="inner").dropna()
    if training.empty:
        raise ValueError("no training rows after joining features and targets")

    target_name = targets.name
    latest_date = features.index.get_level_values("date").max()
    latest_features = features.loc[latest_date].dropna()
    if latest_features.empty:
        raise ValueError("no latest feature rows available")

    training = training.loc[
        training.index.get_level_values("date") < latest_date
    ].tail(train_lookback_rows)
    if training.empty:
        raise ValueError("no training rows before latest feature date")

    x_train = training.drop(columns=[target_name])
    y_train = training[target_name]
    x_latest = latest_features.reindex(columns=x_train.columns)

    estimator = _make_regressor(
        model,
        random_state=random_state,
        training_index=training.index,
    )
    estimator.fit(x_train, y_train)
    predictions = estimator.predict(x_latest)
    return pd.Series(predictions, index=x_latest.index, name=f"mu_{model}")


def _make_regressor(
    model: str,
    *,
    random_state: int,
    training_index: pd.MultiIndex | None = None,
):
    if model == "lasso":
        return make_pipeline(
            StandardScaler(),
            LassoCV(
                cv=_date_ordered_cv_splits(training_index),
                random_state=random_state,
                max_iter=20_000,
            ),
        )
    if model == "elastic_net":
        return make_pipeline(
            StandardScaler(),
            ElasticNetCV(
                cv=_date_ordered_cv_splits(training_index),
                random_state=random_state,
                max_iter=20_000,
            ),
        )
    if model == "random_forest":
        return RandomForestRegressor(
            n_estimators=200,
            max_depth=5,
            min_samples_leaf=10,
            random_state=random_state,
            n_jobs=-1,
        )
    if model == "gradient_boosting":
        return GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=2,
            random_state=random_state,
        )
    raise ValueError(
        "model must be one of: lasso, elastic_net, random_forest, gradient_boosting"
    )


def _date_ordered_cv_splits(
    index: pd.MultiIndex | None,
    *,
    max_splits: int = 5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build expanding-window CV splits that never validate on past dates."""

    if index is None or "date" not in index.names:
        raise ValueError("time-aware CV requires a MultiIndex with a 'date' level")
    if max_splits < 2:
        raise ValueError("max_splits must be at least 2")

    row_dates = pd.Index(index.get_level_values("date"))
    unique_dates = pd.Index(row_dates.unique()).sort_values()
    n_splits = min(max_splits, len(unique_dates) - 1)
    if n_splits < 2:
        raise ValueError("need at least three unique dates for time-aware CV")

    date_blocks = np.array_split(unique_dates.to_numpy(), n_splits + 1)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for split_index in range(1, len(date_blocks)):
        train_dates = np.concatenate(date_blocks[:split_index])
        validation_dates = date_blocks[split_index]
        train_mask = row_dates.isin(train_dates)
        validation_mask = row_dates.isin(validation_dates)
        train_indices = np.flatnonzero(train_mask)
        validation_indices = np.flatnonzero(validation_mask)
        if len(train_indices) and len(validation_indices):
            splits.append((train_indices, validation_indices))

    if len(splits) < 2:
        raise ValueError("need at least two non-empty time-aware CV splits")
    return splits
