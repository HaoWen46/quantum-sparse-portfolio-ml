# Phase 2 Results

Phase 2 goal: move from synthetic QUBO tests to cached ETF data and course-covered ML forecasts.

## Data Cache

Command:

`uv --cache-dir .uv-cache run python scripts/fetch_etf_prices.py`

Result:

- 2,848 daily rows
- 20 ETF tickers
- date range: 2015-01-02 to 2026-04-30
- cache: `data/raw/etf_prices.csv`

Ticker universe:

`SPY, QQQ, IWM, EFA, EEM, TLT, IEF, SHY, LQD, HYG, GLD, SLV, VNQ, DBC, USO, XLK, XLF, XLV, XLE, XLY`

## Tested Code

Command:

`uv --cache-dir .uv-cache run pytest`

Result:

`29 passed`

Coverage now includes:

- cached price loading/saving,
- yfinance close-price extraction,
- return features,
- forward-return target construction,
- sample covariance,
- Ledoit-Wolf shrinkage covariance,
- historical mean forecasts,
- LASSO latest-date forecasts,
- time-ordered CV split construction,
- standard annualized Sharpe computation,
- optional exact enumeration in ML backtests.

## Historical-Mean Baseline Snapshot

Command:

`uv --cache-dir .uv-cache run python scripts/phase2_baseline_selection.py`

Setup:

- latest date: 2026-04-30
- lookback: 252 trading days
- `K = 5`
- covariance: Ledoit-Wolf
- `mu`: trailing historical mean return
- displayed risk aversion: portfolio-level `q = 10`
- binary-selection risk aversion: `q / K = 2`

Result:

| Solver | Selected | Gap vs Exact |
| --- | --- | ---: |
| exact constrained | IWM, EEM, DBC, USO, XLK | 0.00000000 |
| top-K mu | EEM, SLV, DBC, USO, XLK | 0.00068839 |
| greedy | IWM, EEM, DBC, USO, XLK | 0.00000000 |
| local search | IWM, EEM, DBC, USO, XLK | 0.00000000 |
| neal QUBO | IWM, EEM, SLV, DBC, USO | 0.00076570 |

Interpretation:

- top-K expected return is worse under the risk-adjusted objective once covariance/risk is included;
- greedy/local search can match exact on this instance;
- `neal` finds a feasible near-optimal QUBO solution.

## LASSO Forecast Snapshot

Command:

`uv --cache-dir .uv-cache run python scripts/phase2_ml_selection.py`

Setup:

- model: LASSO
- target: 5-day forward return
- features: lagged returns and rolling volatility
- covariance: Ledoit-Wolf
- `K = 5`
- displayed risk aversion: portfolio-level `q = 10`
- binary-selection risk aversion: `q / K = 2`
- LASSO hyperparameter tuning uses expanding, date-ordered CV splits

Result:

| Solver | Selected | Gap vs Exact |
| --- | --- | ---: |
| exact constrained | IEF, SHY, LQD, HYG, DBC | 0.00000000 |
| top-K mu | EEM, SLV, USO, XLV, XLY | 0.00441858 |
| local search | IEF, SHY, LQD, HYG, DBC | 0.00000000 |
| neal QUBO | IEF, HYG, VNQ, DBC, XLV | 0.00042166 |

Interpretation:

- The ML forecast changes the exact sparse basket from the historical-mean baseline.
- Top-K-by-forecast again performs badly under the risk-adjusted objective.
- This supports the project thesis: the interesting CS object is not just forecasting, but the forecast-to-constrained-portfolio optimization layer.

## Next Step

Turn these latest-date snapshots into a walk-forward backtest:

1. pick rebalance schedule, likely monthly;
2. train forecasts using only past data;
3. estimate covariance using trailing returns;
4. solve sparse QUBO selection;
5. hold equal-weight selected assets until next rebalance;
6. compare exact/local/neal/top-K over time.
