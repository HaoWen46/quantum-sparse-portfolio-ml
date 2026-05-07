"""Data loading and caching helpers for ETF price experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf


DEFAULT_ETF_UNIVERSE = (
    "SPY",
    "QQQ",
    "IWM",
    "EFA",
    "EEM",
    "TLT",
    "IEF",
    "SHY",
    "LQD",
    "HYG",
    "GLD",
    "SLV",
    "VNQ",
    "DBC",
    "USO",
    "XLK",
    "XLF",
    "XLV",
    "XLE",
    "XLY",
)


def _extract_close(downloaded: pd.DataFrame) -> pd.DataFrame:
    """Extract adjusted-close-like prices from a yfinance download frame."""

    if downloaded.empty:
        raise ValueError("downloaded price data is empty")

    close_fields = ("Adj Close", "Close")
    if isinstance(downloaded.columns, pd.MultiIndex):
        for field in close_fields:
            if field in downloaded.columns.get_level_values(0):
                return _clean_price_frame(downloaded.xs(field, axis=1, level=0))
            if field in downloaded.columns.get_level_values(1):
                return _clean_price_frame(downloaded.xs(field, axis=1, level=1))
        raise ValueError("could not find Adj Close or Close in yfinance data")

    for field in close_fields:
        if field in downloaded.columns:
            close = downloaded[[field]].copy()
            close.columns = ["price"]
            return _clean_price_frame(close)

    return _clean_price_frame(downloaded)


def _clean_price_frame(prices: pd.DataFrame) -> pd.DataFrame:
    cleaned = prices.copy()
    cleaned.index = pd.to_datetime(cleaned.index)
    cleaned.index.name = None
    cleaned = cleaned.sort_index()
    cleaned.columns = [str(column) for column in cleaned.columns]
    cleaned = cleaned.dropna(axis=0, how="all")
    cleaned = cleaned.dropna(axis=1, how="all")
    if cleaned.empty:
        raise ValueError("price data is empty after cleaning")
    return cleaned


def load_prices_csv(path: str | Path) -> pd.DataFrame:
    """Load cached prices from CSV."""

    prices = pd.read_csv(path, index_col=0, parse_dates=True)
    return _clean_price_frame(prices)


def save_prices_csv(prices: pd.DataFrame, path: str | Path) -> None:
    """Save prices to CSV, creating parent directories if needed."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _clean_price_frame(prices).to_csv(output_path, index_label="date")


def download_prices_yfinance(
    tickers: Iterable[str],
    *,
    start: str,
    end: str | None = None,
) -> pd.DataFrame:
    """Download adjusted close prices from Yahoo Finance via yfinance."""

    ticker_list = list(tickers)
    if not ticker_list:
        raise ValueError("tickers must not be empty")

    downloaded = yf.download(
        ticker_list,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=True,
    )
    close = _extract_close(downloaded)
    return close.reindex(columns=ticker_list).dropna(axis=1, how="all")


def get_prices(
    tickers: Iterable[str],
    *,
    start: str,
    end: str | None = None,
    cache_path: str | Path = "data/raw/etf_prices.csv",
    refresh: bool = False,
) -> pd.DataFrame:
    """Load cached prices, or download and cache them if needed."""

    path = Path(cache_path)
    if path.exists() and not refresh:
        cached = load_prices_csv(path)
        requested = [ticker for ticker in tickers if ticker in cached.columns]
        if requested:
            return cached[requested]

    prices = download_prices_yfinance(tickers, start=start, end=end)
    save_prices_csv(prices, path)
    return prices


def returns_from_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute simple daily returns from price levels."""

    return _clean_price_frame(prices).pct_change(fill_method=None).dropna(how="all")
