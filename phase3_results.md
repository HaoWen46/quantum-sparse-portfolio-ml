# Phase 3 Results

Phase 3 goal: move from latest-date snapshots to walk-forward monthly backtests.

## Verification Status

Command:

`uv --cache-dir .uv-cache run pytest`

Result:

`29 passed`

New coverage:

- monthly rebalance date generation,
- walk-forward historical-mean backtest,
- equal-weight portfolio return construction,
- standard annualized Sharpe summary metrics,
- selection/turnover tracking.

## Historical-Mean Backtest

Command:

`uv --cache-dir .uv-cache run python scripts/phase3_backtest_historical.py`

Setup:

- ETF universe: 20 ETFs
- period: monthly rebalances from 2018 onward
- forecast: trailing 252-day historical mean
- covariance: Ledoit-Wolf
- `K = 5`
- displayed risk aversion: portfolio-level `q = 10`
- binary-selection risk aversion: `q / K = 2`
- QUBO penalty multiplier: `0.1`
- solvers: exact, top-K, local search, neal

Output files:

- `results/phase3/historical_daily_returns.csv`
- `results/phase3/historical_summary.csv`
- `results/phase3/historical_selections.csv`

Summary:

| Strategy | Annual Return | Annual Volatility | Sharpe | Max Drawdown | Mean Turnover | Mean Objective Gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| exact | 0.1082 | 0.1233 | 0.8954 | -0.1719 | 0.2020 | 0.000000 |
| local_search | 0.1063 | 0.1229 | 0.8842 | -0.1719 | 0.2061 | 0.000001 |
| top_k_mu | 0.1501 | 0.1820 | 0.8601 | -0.2538 | 0.2102 | 0.002604 |
| neal | 0.1063 | 0.1304 | 0.8406 | -0.2196 | 0.6388 | 0.001453 |

Interpretation:

- Top-K by forecast has higher raw return, but much higher volatility and drawdown.
- Exact/local risk-aware portfolios have better Sharpe and lower drawdown.
- Local search almost exactly matches exact optimization, which suggests a strong classical heuristic baseline.
- `neal` is feasible but noisier, with higher turnover and lower realized performance in this configuration.

## LASSO Forecast Backtest

Command:

`uv --cache-dir .uv-cache run python scripts/phase3_backtest_lasso.py`

Setup:

- ETF universe: 20 ETFs
- period: monthly rebalances from 2020 onward
- forecast: LASSO on lagged-return and volatility features
- target: 5-day forward return
- covariance: Ledoit-Wolf
- `K = 5`
- displayed risk aversion: portfolio-level `q = 10`
- binary-selection risk aversion: `q / K = 2`
- LASSO hyperparameter tuning uses expanding, date-ordered CV splits
- solvers: exact, top-K, local search

Output files:

- `results/phase3/lasso_daily_returns.csv`
- `results/phase3/lasso_summary.csv`
- `results/phase3/lasso_selections.csv`

Summary:

| Strategy | Annual Return | Annual Volatility | Sharpe | Max Drawdown | Mean Turnover | Mean Objective Gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| top_k_mu | 0.2294 | 0.2002 | 1.1320 | -0.2050 | 0.5459 | 0.007102 |
| exact | 0.0769 | 0.1008 | 0.7855 | -0.1604 | 0.2892 | 0.000000 |
| local_search | 0.0726 | 0.1011 | 0.7439 | -0.1604 | 0.2892 | 0.000001 |

Interpretation:

- LASSO top-K wins on realized Sharpe in this initial run, despite being worse under the in-sample risk-adjusted objective.
- This is not a failure; it is a useful research result. The QUBO optimizer is doing exactly what it was asked to do: reduce the risk-adjusted objective. That objective may still be too conservative or too sensitive to covariance/risk-aversion choices.
- The later q/K sweep and gradient boosting robustness run address this directly: QUBO helps in some risk/cardinality regimes, but top-K remains competitive when the forecast signal rewards taking more volatility.

## Research Takeaway So Far

The core project is shaping into:

> QUBO sparse optimization is easy to formulate correctly, but its realized financial behavior depends strongly on the risk objective, penalty scaling, and solver noise. The right question is not whether quantum-inspired optimization magically beats top-K ML ranking; it is when the risk-aware binary optimizer gives a better risk-return tradeoff than simple ranking baselines.

That is a much stronger final-project story than a one-sided "quantum wins" demo.

## Gradient Boosting Robustness Backtest

Command:

`uv --cache-dir .uv-cache run python scripts/phase3_backtest_gradient_boosting.py`

Setup:

- ETF universe: 20 ETFs
- period: monthly rebalances from 2020 onward
- forecast: gradient boosting on lagged-return and volatility features
- target: 5-day forward return
- covariance: Ledoit-Wolf
- `K = 5`
- displayed risk aversion: portfolio-level `q = 10`
- binary-selection risk aversion: `q / K = 2`
- solvers: exact, top-K, local search

Output files:

- `results/phase3/gradient_boosting_daily_returns.csv`
- `results/phase3/gradient_boosting_summary.csv`
- `results/phase3/gradient_boosting_selections.csv`

Summary:

| Strategy | Annual Return | Annual Volatility | Sharpe | Max Drawdown | Mean Turnover | Mean Objective Gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| top_k_mu | 0.2004 | 0.1889 | 1.0619 | -0.3103 | 0.6757 | 0.001212 |
| exact | 0.1551 | 0.1750 | 0.9116 | -0.3070 | 0.6730 | 0.000000 |
| local_search | 0.1551 | 0.1750 | 0.9116 | -0.3070 | 0.6730 | 0.000000 |

Interpretation:

- Top-K still wins realized Sharpe, but exact/local QUBO portfolios are much closer to top-K than in the LASSO backtest.
- Exact and local search are identical in this run, which reinforces local search as a strong practical optimizer.
- This robustness check supports the broader story: QUBO controls the risk-return objective, but realized out-of-sample performance depends heavily on forecast behavior.

## Historical q/K Sweep

Command:

`uv --cache-dir .uv-cache run python scripts/phase3_sweep_historical.py`

Output:

- `results/phase3/historical_q_k_sweep.csv`

This sweep skips repeated exact enumeration and compares the two fast decision rules:

- `top_k_mu`: choose the assets with highest expected return forecast;
- `local_search`: risk-aware cardinality-preserving optimizer.

Sweep grid:

- portfolio-level `q in {1, 3, 10, 30}`
- `K in {3, 5, 8}`

Key patterns:

- For `K = 3`, top-K wins Sharpe for all tested `q`; risk-aware selection lowers drawdown as `q` rises, but gives up too much return.
- For `K = 5`, local search wins Sharpe at `q = 10` and `q = 30`; the best historical-mean Sharpe in this grid is `K = 5, q = 30`.
- For `K = 8`, local search wins Sharpe at `q = 10`; top-K wins at `q = 1, 3, 30`.
- High risk aversion can help drawdown-adjusted performance in a moderate-cardinality basket, but it can also become too conservative for larger baskets.

Selected rows:

| K | q | Strategy | Annual Return | Annual Volatility | Sharpe | Max Drawdown |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 3 | 1 | top_k_mu | 0.1968 | 0.2257 | 0.9100 | -0.2935 |
| 3 | 30 | local_search | 0.0681 | 0.0788 | 0.8761 | -0.1498 |
| 5 | 10 | local_search | 0.1063 | 0.1229 | 0.8842 | -0.1719 |
| 5 | 10 | top_k_mu | 0.1501 | 0.1820 | 0.8601 | -0.2538 |
| 5 | 30 | local_search | 0.0782 | 0.0806 | 0.9742 | -0.1273 |
| 8 | 10 | local_search | 0.1017 | 0.1141 | 0.9065 | -0.1470 |
| 8 | 10 | top_k_mu | 0.1340 | 0.1609 | 0.8622 | -0.2506 |

Interpretation:

The optimizer helps most when the portfolio has enough assets for diversification and the risk-aversion setting meaningfully penalizes covariance. In this corrected equal-weight interpretation, the strongest historical-mean improvement appears at `K = 5, q = 30`, with another clear win at `K = 8, q = 10`. If `K` is too small, ranking dominates.

Optional next sweeps:

- run a smaller LASSO q/K sweep;
- add a turnover penalty and test whether `neal` stabilizes;
- add presentation-deck formatting around the generated figures.

The presentation can now show regimes where QUBO helps and regimes where ranking is enough. The first figure set is documented in `phase4_figures.md` and generated under `results/figures/`.
