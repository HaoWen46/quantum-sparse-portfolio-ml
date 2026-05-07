# Phase 0 Research Log

Phase 0 goal: lock the research framing, formulas, source base, library candidates, dataset plan, and validation requirements before implementation.

## Non-Negotiable Execution Rule

Use `uv` whenever Python is involved.

Allowed command patterns:

- `uv init`
- `uv add ...`
- `uv sync`
- `uv run python ...`
- `uv run pytest`

Avoid:

- bare `python`
- bare `pip`
- bare `pytest`

This matters both because of the workspace rule and because a reproducible final project environment is part of the deliverable.

## Current Project Thesis

We combine course-covered ML predictors with a QUBO formulation of sparse portfolio selection, then benchmark exact, greedy, local-search, and annealing-style solvers under realistic walk-forward validation.

The project is about algorithmic formulation and solver behavior. It is not about claiming quantum advantage or pretending stock-return prediction is easy.

## Locked MVP

Use the binary portfolio-selection problem:

`min_x q x' Sigma x - mu' x + A (1' x - K)^2`

with:

- `x_i in {0,1}`,
- `K` selected assets,
- equal weight `w_i = x_i / K` after selection,
- `mu` from ML forecasts or simple baselines,
- `Sigma` from trailing returns.

Why this MVP is right:

- It is already quadratic.
- It maps cleanly to QUBO/BQM.
- It has a known exact small-instance solution by enumeration.
- It matches Qiskit Finance's educational portfolio objective.
- It is easier to explain than continuous weight encodings.

## Formula Checks

With upper-triangular QUBO convention:

`E(x) = sum_{i <= j} Q_ij x_i x_j + const`

and symmetric `Sigma`:

`Q_ii = q Sigma_ii - mu_i + A(1 - 2K)`

`Q_ij = 2q Sigma_ij + 2A`, for `i < j`.

Constant offset:

`A K^2`.

Validation requirement:

- Direct energy and QUBO energy must agree up to the constant offset.
- Exact constrained optimum among `sum x = K` must match QUBO optimum for sufficiently large `A`.
- Penalty sweep must show when infeasible portfolios stop winning.

## Source Quality Notes

High-confidence sources:

- Qiskit Finance portfolio tutorial: clean binary mean-variance formulation.
- Sakuler et al. 2025: real-world QUBO/annealing portfolio test; strong source for caution around penalty tuning.
- D-Wave `dimod` docs: BQM representation.
- D-Wave `neal` docs: local simulated annealing sampler.
- scikit-learn LedoitWolf docs: covariance shrinkage formula and implementation.
- Lucas 2014: general Ising/QUBO formulation reference.

Useful but secondary:

- Lu et al. 2024: directly quantum-inspired portfolio optimization, but arXiv/RePEc rather than a mature journal source.
- Palmer et al. 2021/2022: good for target-volatility, investment bands, and cardinality constraints.
- D-Wave portfolio example: useful implementation reference, but CQM/DQM-heavy and cloud-solver-oriented.

## Library Decision

Phase 1 should not depend on quantum packages at first.

Initial implementation:

- `numpy`
- `pandas`
- `scikit-learn`

Then try:

- `dimod` for BQM representation and exact solver,
- `dwave-neal` or `dwave-samplers` for simulated annealing,
- OpenJij as optional simulated quantum annealing comparison.

Avoid early:

- Qiskit Finance as a dependency,
- D-Wave cloud solvers,
- PyQUBO unless constraints become more complex,
- real quantum hardware.

Reason:

The project should be able to run locally and reproducibly. Quantum libraries are supporting characters, not the spine.

## Dataset Plan

Preferred universe:

- Liquid ETFs first, because they reduce single-stock idiosyncratic weirdness.
- Candidate tickers: `SPY`, `QQQ`, `IWM`, `EFA`, `EEM`, `TLT`, `IEF`, `SHY`, `LQD`, `HYG`, `GLD`, `SLV`, `VNQ`, `DBC`, `USO`, `XLK`, `XLF`, `XLV`, `XLE`, `XLY`.

Why ETFs:

- Easier covariance estimation.
- Cleaner asset-class interpretation.
- Less need for survivorship-bias discussion than hand-picking individual stocks.
- Works well with the "sparse allocation" story.

Possible stock universe:

- Large, liquid US names for a secondary test.
- Keep stocks as optional because survivorship bias and corporate actions are noisier.

Data source:

- Use `yfinance` for educational/research data if available.
- Cache raw adjusted close data as CSV.
- Never redownload inside every experiment.

## ML Plan

Course-compatible models:

- historical mean baseline,
- LASSO,
- elastic net,
- random forest,
- gradient boosting.

Forecast target:

- Start with next-period return regression for simplicity.
- Also test cross-sectional ranking/classification if regression forecasts are too noisy.

Feature set:

- trailing returns,
- rolling volatility,
- rolling momentum,
- moving-average distance,
- drawdown,
- optional market ETF lag features.

Validation:

- walk-forward only.
- no random K-fold for financial time series.
- transformations must be fit on training windows only.

## Solver Benchmark Plan

Core solvers:

1. Exact constrained enumeration for small `n`.
2. Exact full-QUBO enumeration for small `n`.
3. Top-K by `mu`.
4. Greedy marginal risk-return selection.
5. Cardinality-preserving local search by swaps.
6. Simulated annealing over QUBO.
7. Optional `dimod`/`neal`/OpenJij.

Core solver metrics:

- feasibility rate,
- objective value,
- objective gap vs exact optimum,
- runtime,
- sensitivity to `A`,
- stability over random seeds.

Finance metrics:

- return,
- volatility,
- Sharpe,
- max drawdown,
- turnover,
- selection heatmap.

## Open Questions

1. Should `mu` be raw predicted return or a cross-sectional rank score?
   - Current answer: implement both if time allows; start with raw return for formula clarity.

2. Should the MVP use ETFs or stocks?
   - Current answer: ETFs first.

3. Should continuous weights be included?
   - Current answer: only as a two-stage extension after binary selection works.

4. Should we include real quantum algorithms?
   - Current answer: no, unless everything else is finished.

5. Should we include index tracking?
   - Current answer: keep as fallback if return forecasting looks too noisy.

## Phase 0 Exit Criteria

Phase 0 is done when:

- formulas are documented,
- key papers are listed,
- library stack is chosen,
- dataset candidate universe is chosen,
- validation rules are explicit,
- Phase 1 scaffold plan is ready.

Current status: complete. The `uv` scaffold exists, and core packages install cleanly with a project-local uv cache.

## Immediate Next Step

Phase 1 started with a small local `uv` project:

1. `uv --cache-dir .uv-cache init --bare ...`
2. `uv --cache-dir .uv-cache add numpy pandas scikit-learn pytest`
3. create `src/` package for QUBO builder and exact verifier
4. create tests for the QUBO coefficient expansion
5. only then add optional annealing packages
