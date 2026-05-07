import numpy as np
import pandas as pd

from sparse_portfolio import (
    ledoit_wolf_covariance,
    load_prices_csv,
    make_return_feature_panel,
    returns_from_prices,
    sample_covariance,
    save_prices_csv,
)
from sparse_portfolio.data import _extract_close


def _prices() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    return pd.DataFrame(
        {
            "SPY": np.linspace(100, 120, len(dates)),
            "TLT": np.linspace(80, 82, len(dates)),
            "GLD": np.linspace(150, 165, len(dates)),
        },
        index=dates,
    )


def test_extract_close_from_yfinance_multiindex():
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    columns = pd.MultiIndex.from_product([["Close", "Volume"], ["SPY", "TLT"]])
    downloaded = pd.DataFrame(
        [
            [100.0, 80.0, 1_000, 2_000],
            [101.0, 81.0, 1_100, 2_100],
            [102.0, 82.0, 1_200, 2_200],
        ],
        index=dates,
        columns=columns,
    )

    close = _extract_close(downloaded)

    assert close.columns.tolist() == ["SPY", "TLT"]
    assert close.iloc[0].tolist() == [100.0, 80.0]


def test_price_csv_roundtrip(tmp_path):
    prices = _prices()
    path = tmp_path / "prices.csv"

    save_prices_csv(prices, path)
    loaded = load_prices_csv(path)

    pd.testing.assert_frame_equal(loaded, prices, check_freq=False)


def test_returns_from_prices():
    returns = returns_from_prices(_prices())

    assert returns.shape[0] == 29
    assert set(returns.columns) == {"SPY", "TLT", "GLD"}


def test_make_return_feature_panel_has_expected_columns():
    panel = make_return_feature_panel(
        _prices(),
        return_windows=(1, 5),
        volatility_windows=(5,),
        target_horizon=3,
    )

    assert {"ret_1d", "ret_5d", "vol_5d", "fwd_ret_3d"}.issubset(panel.columns)
    assert panel.index.names == ["date", "ticker"]


def test_covariance_estimators_return_square_matrices():
    returns = returns_from_prices(_prices())

    sample = sample_covariance(returns)
    shrink = ledoit_wolf_covariance(returns)

    assert sample.shape == (3, 3)
    assert shrink.shape == (3, 3)
    assert np.allclose(shrink, shrink.T)
