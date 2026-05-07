# Strategy And Plan

## Project Shape

Build a research-grade mini pipeline:

`market data -> ML forecast -> QUBO portfolio selection -> solver benchmark -> walk-forward backtest`

The key is to make the optimizer and validation rigorous enough that the presentation feels like a CS systems/algorithms project, while still satisfying the course requirement through the ML forecast models.

## Phase 0: Research And Design

Status: complete.

Tasks:

- [x] Collect key QUBO/portfolio/quantum-inspired papers.
- [x] Lock the MVP formulation.
- [x] Decide initial solver stack.
- [x] Decide candidate dataset universe.
- [x] Write formulas and validation checks.
- [x] Create the Phase 1 `uv` project scaffold.
- [x] Confirm package installation under `uv`.

Deliverable:

- `research_brief.md`
- `math_formulation.md`
- this plan
- `phase_0_research_log.md`

## Phase 1: Minimal Reproducible QUBO

Status: complete.

Completed:

- [x] Project-local `uv` scaffold.
- [x] Core dependencies installed through `uv`.
- [x] QUBO coefficient builder.
- [x] Direct penalized objective and QUBO energy evaluator.
- [x] Exact constrained enumeration.
- [x] Exact full-QUBO enumeration.
- [x] Formula tests.
- [x] Deterministic toy demo script.

Remaining:

- [x] Add greedy baseline.
- [x] Add cardinality-preserving local search.
- [x] Add simple simulated annealing baseline.
- [x] Run penalty sweeps on generated random instances.
- [x] Decide whether to add `dimod`/`neal`.

Build a toy notebook or script that takes `mu`, `Sigma`, `K`, `q`, and `A`, then outputs:

- QUBO coefficients,
- exact optimum by enumeration for `n <= 20` if feasible,
- greedy baseline,
- simulated annealing samples.

Validation:

- unit test QUBO expansion against direct energy computation,
- compare exact constrained optimum to QUBO optimum under large enough `A`,
- sweep `A` and show feasibility behavior.

Recommended implementation:

- Use `uv` for all Python commands. No bare `python`, `pip`, or `pytest`.
- Start with only `numpy`, `pandas`, `scikit-learn`.
- Add `dimod`/`neal` if dependency installation is smooth.
- If annealing dependencies fail, implement a simple local-search/simulated annealing solver ourselves. That is still quantum-inspired enough if positioned carefully.

## Phase 2: Forecasting Models

Status: complete for MVP.

Completed:

- [x] Add cached ETF price data utilities.
- [x] Add return feature panel builder.
- [x] Add sample and Ledoit-Wolf covariance helpers.
- [x] Fetch/cache ETF prices.
- [x] Create baseline historical-mean forecasts.
- [x] Add LASSO / elastic net forecaster.
- [x] Add random forest / gradient boosting forecasters.
- [x] Connect forecasts and covariance estimates to QUBO selection.

Remaining:

- [ ] Optional: tune classification/ranking targets as a robustness extension.

Use course-covered ML methods:

- baseline: historical mean returns,
- LASSO / elastic net,
- random forest,
- gradient boosting.

Feature candidates:

- lagged returns: 1d, 5d, 20d,
- rolling volatility,
- rolling momentum,
- rolling drawdown,
- moving-average distance,
- volume change if available,
- market/sector ETF lag features.

Prediction target options:

- regression: next `h`-day return,
- classification: whether next `h`-day return is above cross-sectional median,
- ranking score: predicted outperformance probability.

Recommendation:

Use classification/ranking for robustness, then map score into `mu`. Raw return regression is very noisy.

## Phase 3: Backtest Design

Status: complete for MVP.

Completed:

- [x] Add monthly rebalance date generation.
- [x] Add walk-forward historical-mean backtest.
- [x] Add performance summary metrics.
- [x] Add ML/LASSO walk-forward variant.

Remaining:

- [x] Run and document ETF historical-mean backtest.
- [x] Run and document ETF LASSO backtest.
- [x] Run and document risk-aversion/cardinality sweeps.
- [x] Add presentation plots.

Use walk-forward validation:

- train on past window,
- predict next rebalance period,
- optimize portfolio,
- hold for `h` days,
- rebalance.

Avoid:

- random K-fold on time series,
- feature normalization using future data,
- covariance estimated using test period.

Portfolio configurations:

- equal-weight selected portfolio,
- optional two-stage continuous weights over selected assets,
- equal-weight top-K by forecast as a non-QUBO baseline,
- minimum variance selected basket as risk baseline.

Metrics:

- annualized return,
- annualized volatility,
- Sharpe ratio,
- max drawdown,
- turnover,
- average cardinality violation before filtering,
- solver runtime,
- exact optimality gap on small subproblems.

## Phase 4: Solver Benchmark Extensions

Solvers to compare:

1. Exact enumeration for small `n`.
2. Top-K by predicted return.
3. Greedy risk-return selection.
4. Local search with cardinality-preserving swaps.
5. Simulated annealing over full QUBO.
6. Optional: `neal`, OpenJij, or D-Wave Ocean hybrid tooling.

Important:

Benchmark solvers on the same generated QUBO instances before doing finance backtests. This separates optimizer quality from forecast noise.

Status: complete for first deck draft.

Completed:

- [x] Exact, greedy, local-search, in-house annealing, `dimod`, and `neal` comparisons on a toy synthetic instance.
- [x] Penalty-feasibility sweeps on generated synthetic instances.
- [x] Runtime and objective-gap scaling across synthetic instances with increasing `n`.
- [x] Solver runtime and gap scaling plots.

Remaining:

- [ ] Optional: repeat scaling with larger `K` or more seeds if the deck needs a stronger solver benchmark.

## Phase 5: Presentation Story

Suggested story:

1. Course methods produce forecasts: LASSO/RF/GB.
2. Finance objective asks for return-risk tradeoff.
3. Sparse portfolios introduce a combinatorial constraint.
4. QUBO converts that constraint into binary energy minimization.
5. Annealing-style solvers search low-energy portfolios.
6. Results show where the method works, where penalty tuning matters, and whether the solver gives anything over simple baselines.

Best figures:

- QUBO energy decomposition diagram.
- Penalty sweep: feasibility rate vs `A`.
- Solver runtime / optimality gap curve.
- Backtest equity curve.
- Risk-return scatter of portfolio strategies.
- Heatmap of selected assets across rebalance dates.

## Phase 4: Presentation Figures

Status: complete for first deck draft.

Completed:

- [x] Add reproducible Matplotlib figure-generation script.
- [x] Plot penalty-feasibility threshold.
- [x] Plot historical-mean and LASSO equity curves.
- [x] Plot q/K Sharpe heatmap and local-search minus top-K Sharpe delta.
- [x] Plot local-search selection frequency.
- [x] Document how each figure supports the presentation story.

Generated figures:

- `results/figures/penalty_feasibility.png`
- `results/figures/historical_equity_curves.png`
- `results/figures/lasso_equity_curves.png`
- `results/figures/historical_risk_return_scatter.png`
- `results/figures/lasso_risk_return_scatter.png`
- `results/figures/historical_qk_sharpe_heatmap_local_search.png`
- `results/figures/historical_qk_sharpe_delta.png`
- `results/figures/historical_selection_frequency_local_search.png`
- `results/figures/solver_runtime_scaling.png`
- `results/figures/solver_gap_scaling.png`

## Scope Control

Must-have:

- Correct QUBO math.
- Exact validation on small instances.
- At least one ML model from course slides.
- Walk-forward backtest.
- Solver comparison.

Nice-to-have:

- OpenJij or D-Wave Ocean sampler.
- Two-stage continuous weighting.
- Turnover penalty.
- Index tracking variant.

Avoid unless everything else is done:

- Real quantum hardware.
- QAOA/VQE demos.
- Deep learning.
- Complex transaction-cost model.
- Overclaiming investment performance.

## Concrete Next Step

Build the actual slide deck from `presentation_outline.md`. If there is time for one more coding pass before the deck, add one robustness ML backtest with gradient boosting.
