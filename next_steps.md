# Next Steps

This project is past the MVP line. The remaining work should make the presentation more convincing, not expand the project until it becomes mushy.

## Priority 1: Build The Deck

Create the actual presentation from `presentation_outline.md`.

Use these figures first:

- `results/figures/penalty_feasibility.png`
- `results/figures/historical_equity_curves.png`
- `results/figures/historical_risk_return_scatter.png`
- `results/figures/lasso_equity_curves.png`
- `results/figures/historical_qk_sharpe_delta.png`
- `results/figures/historical_selection_frequency_local_search.png`

Deck structure:

1. project question,
2. pipeline,
3. QUBO formulation,
4. penalty validation,
5. solver scaling,
6. why quantum-inspired matters,
7. historical-mean backtest,
8. LASSO counterexample,
9. gradient boosting robustness,
10. q/K regime analysis,
11. interpretation and limitations.

The deck should emphasize a defensible claim:

> QUBO sparse optimization is useful as a controllable decision layer, but it does not universally beat simple forecast ranking.

## Priority 2: Use Solver Runtime Evidence In The Deck

Status: first pass complete.

Generated files:

- `results/phase5/solver_scaling.csv`
- `results/phase5/solver_scaling_summary.csv`
- `results/figures/solver_runtime_scaling.png`
- `results/figures/solver_gap_scaling.png`

Main talking points:

- exact full-QUBO enumeration scales poorly and was only run through `n = 16`;
- exact constrained enumeration remains useful as a small-instance correctness oracle;
- local search stays very fast and nearly exact in the synthetic benchmark;
- simple annealing baselines are feasible but have larger objective gaps in this configuration.

## Priority 3: Use The Robustness ML Run

Status: first pass complete.

Generated files:

- `results/phase3/gradient_boosting_daily_returns.csv`
- `results/phase3/gradient_boosting_summary.csv`
- `results/phase3/gradient_boosting_selections.csv`
- `results/figures/gradient_boosting_equity_curves.png`
- `results/figures/gradient_boosting_risk_return_scatter.png`

Main talking points:

- top-K still wins realized Sharpe;
- exact/local QUBO portfolios are much closer to top-K than in the LASSO run;
- exact and local search paths overlap, which is good evidence that local search is a strong practical optimizer.

## Priority 4: Consider A Turnover Penalty

`neal` has high turnover in the historical backtest. A turnover penalty is already QUBO-friendly:

`C sum_i (x_i - x_prev_i)^2`.

This adds a linear diagonal adjustment:

`Q_ii += C (1 - 2 x_prev_i)`.

This is a good extension if there is time because it is clearly financial, clearly QUBO-native, and likely improves the noisy annealing story.

Deliverables:

- turnover-aware local search/QUBO option,
- turnover penalty sweep,
- before/after table for mean turnover and Sharpe.

## Priority 5: Polish The Research Narrative

Before final submission, tighten these points:

- explain that `q` in tables is portfolio-level risk aversion and the QUBO uses `q / K`;
- state that Sharpe is annualized mean return over annualized volatility;
- state that LASSO CV uses expanding date-ordered splits;
- avoid any "quantum advantage" phrasing;
- call `neal` an annealing-style baseline, not evidence of quantum superiority.
- explain quantum future potential as a decision-layer story: quantum/hybrid solvers may matter for large constrained binary search, not for magically forecasting returns.
- include the ECON takeaway: econometrics estimates the inputs, ML forecasts the signals, and optimization turns them into decisions.

## Do Not Do Unless Everything Else Is Finished

- real quantum hardware,
- QAOA/VQE,
- deep learning,
- large stock universes,
- transaction-cost realism beyond a simple turnover penalty,
- more than one additional ML robustness model.
