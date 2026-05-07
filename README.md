# Quantum-Inspired Sparse Portfolio Optimization

Goal: build a CS-forward final project around financial ML, sparse portfolio selection, and QUBO/annealing-style optimization.

Course hook: the project uses ML methods covered in ECON7225, especially LASSO/ridge/elastic net, random forests, gradient boosting, cross-validation, and the Financial Machine Learning unit. The optimization layer is the CS artifact: convert ML forecasts and risk estimates into a sparse portfolio-selection QUBO, then benchmark exact, greedy, local-search, and annealing-style solvers.

Presentation date: June 9, 2026.

## Current Thesis

The safest and most interesting project is not "quantum computers beat classical finance." It is:

> Given ML forecasts of asset returns/risk, can a QUBO-based sparse optimizer produce competitive constrained portfolios, and how sensitive is it to penalty scaling, solver choice, and validation design?

That framing is honest, implementation-friendly, and flexible enough to include quantum-inspired algorithms without needing actual quantum hardware.

## Folder Map

- `research_brief.md`: papers, libraries, and source takeaways.
- `math_formulation.md`: portfolio objective, QUBO construction, constraints, and validation checks.
- `strategy_plan.md`: implementation strategy, experiments, and presentation plan.
- `phase_0_research_log.md`: Phase 0 decisions, open questions, and immediate next steps.
- `phase1_results.md`: current QUBO validation and synthetic solver results.
- `phase2_results.md`: cached ETF data and latest-date forecast/selection results.
- `phase3_results.md`: monthly walk-forward backtest results.
- `phase4_figures.md`: generated presentation figures and slide-use notes.
- `presentation_outline.md`: slide-by-slide story built from the current results.
- `report.md`: full written report draft.
- `next_steps.md`: prioritized work remaining before the final presentation.
- `references.bib`: starting bibliography.

## Recommended MVP

1. Build daily-return dataset for 20-50 liquid ETFs/stocks.
2. Train course-covered forecasting models with walk-forward validation:
   - LASSO / elastic net
   - random forest
   - gradient boosting
3. At each rebalance date, estimate expected returns μ and covariance Σ.
4. Solve sparse selection:

```math
\min_x \; q x^\top \Sigma x - \mu^\top x + A\left(\sum_i x_i - K\right)^2,
\qquad x_i \in \{0,1\}.
```

5. Benchmark solvers:
   - exact brute force for small n
   - greedy/local-search baseline
   - simulated annealing / quantum-inspired sampler
   - optionally D-Wave Ocean or OpenJij if installation is painless
6. Evaluate:
   - out-of-sample return, volatility, Sharpe
   - turnover
   - feasibility rate
   - objective gap vs exact small-instance optimum
   - runtime and scaling

## Current Code

Phase 1 has started with a small tested package:

- `src/sparse_portfolio/qubo.py`: QUBO construction, energy evaluation, and exact enumeration.
- `src/sparse_portfolio/solvers.py`: top-K, greedy, swap local search, simple simulated annealing, `dimod`, and `neal` baselines.
- `src/sparse_portfolio/experiments.py`: synthetic instance generation, penalty sweeps, and solver benchmark tables.
- `tests/test_qubo.py` and `tests/test_solvers.py`: formula, exact-solver, and baseline tests.
- `scripts/phase1_toy_qubo.py`: deterministic toy demo with a penalty sweep.
- `scripts/phase1_random_sweep.py`: random-instance sweeps that write CSVs under `results/phase1/`.
- `scripts/fetch_etf_prices.py`: downloads and caches the default ETF universe.
- `scripts/phase2_baseline_selection.py`: latest-date historical-mean forecast selection.
- `scripts/phase2_ml_selection.py`: latest-date ML forecast selection.
- `scripts/phase3_backtest_historical.py`: monthly walk-forward historical-mean backtest.
- `scripts/phase3_backtest_lasso.py`: monthly walk-forward LASSO forecast backtest.
- `scripts/phase3_backtest_gradient_boosting.py`: monthly walk-forward gradient boosting forecast backtest.
- `scripts/phase3_sweep_historical.py`: q/K sweep for historical-mean backtests.
- `scripts/phase4_make_figures.py`: creates presentation-ready PNG figures from generated CSVs.
- `scripts/phase5_solver_scaling.py`: synthetic solver runtime/objective-gap scaling benchmark.
- `results/phase1/`: generated Phase 1 CSV outputs.
- `results/phase5/`: generated solver scaling CSV outputs.
- `results/figures/`: generated presentation figures.

Useful commands:

- `uv --cache-dir .uv-cache sync`
- `uv --cache-dir .uv-cache run pytest`
- `uv --cache-dir .uv-cache run python scripts/phase1_toy_qubo.py`
- `uv --cache-dir .uv-cache run python scripts/phase1_random_sweep.py`
- `uv --cache-dir .uv-cache run python scripts/fetch_etf_prices.py`
- `uv --cache-dir .uv-cache run python scripts/phase2_baseline_selection.py`
- `uv --cache-dir .uv-cache run python scripts/phase2_ml_selection.py`
- `uv --cache-dir .uv-cache run python scripts/phase3_backtest_historical.py`
- `uv --cache-dir .uv-cache run python scripts/phase3_backtest_lasso.py`
- `uv --cache-dir .uv-cache run python scripts/phase3_backtest_gradient_boosting.py`
- `uv --cache-dir .uv-cache run python scripts/phase3_sweep_historical.py`
- `uv --cache-dir .uv-cache run python scripts/phase5_solver_scaling.py`
- `uv --cache-dir .uv-cache run python scripts/phase4_make_figures.py`

## Working Rule

All Python work in this repo should run through `uv`, matching the course/workspace instruction.

Use patterns like:

- `uv init`
- `uv add numpy pandas scikit-learn`
- `uv run python ...`
- `uv run pytest`

Do not use bare `python`, `pip`, or `pytest` commands for this project.
