# Phase 4 Figures

Phase 4 goal: convert the validated experiments into presentation-ready figures.

## Command

`uv --cache-dir .uv-cache run python scripts/phase4_make_figures.py`

## Output Files

- `results/figures/penalty_feasibility.png`
- `results/figures/historical_equity_curves.png`
- `results/figures/lasso_equity_curves.png`
- `results/figures/gradient_boosting_equity_curves.png`
- `results/figures/historical_risk_return_scatter.png`
- `results/figures/lasso_risk_return_scatter.png`
- `results/figures/gradient_boosting_risk_return_scatter.png`
- `results/figures/historical_qk_sharpe_heatmap_local_search.png`
- `results/figures/historical_qk_sharpe_delta.png`
- `results/figures/historical_selection_frequency_local_search.png`
- `results/figures/solver_runtime_scaling.png`
- `results/figures/solver_gap_scaling.png`

## Figure Roles

### Penalty Feasibility

`penalty_feasibility.png` shows why penalty calibration matters. In the synthetic sweep, the QUBO cardinality penalty becomes reliable at a multiplier of about `0.1`; below that, the unconstrained QUBO optimum can violate the target portfolio size.

Presentation use:

- introduce the constrained-to-QUBO conversion,
- show that `A` is not a harmless implementation detail,
- justify why the later experiments use a validated penalty range.

### Equity Curves

`historical_equity_curves.png` shows the central historical-mean result:

- top-K by forecast earns more raw return,
- exact/local QUBO portfolios have smoother paths,
- local search closely tracks exact enumeration,
- `neal` is feasible but noisier in this setup.

`lasso_equity_curves.png` is the useful counterexample:

- LASSO top-K wins realized growth and Sharpe,
- exact/local QUBO still minimize the specified risk-adjusted objective,
- this motivates treating the risk-aversion parameter as a modeling choice rather than a fixed truth.

`gradient_boosting_equity_curves.png` is the nonlinear ML robustness check:

- top-K still wins realized Sharpe,
- exact/local QUBO portfolios are closer to top-K than under LASSO,
- exact and local search overlap, showing local search recovered the exact constrained path.

### Risk-Return Scatter

`historical_risk_return_scatter.png` and `lasso_risk_return_scatter.png` show annual return against annual volatility, with Sharpe labels.

Presentation use:

- show that top-K often buys return with much more volatility,
- show that QUBO/local-search portfolios are lower-volatility alternatives,
- make the return/risk tradeoff visible without asking the audience to parse a table.

### q/K Heatmaps

`historical_qk_sharpe_heatmap_local_search.png` summarizes the local-search optimizer over risk aversion `q` and cardinality `K`.

`historical_qk_sharpe_delta.png` is the strongest presentation figure. It shows where risk-aware local search beats top-K:

- local search helps most at `K = 5`, especially `q = 10` and `q = 30`;
- local search is also useful at `K = 8` for `q = 10`;
- top-K dominates at `K = 3`;
- very high risk aversion can help a `K = 5` defensive basket, but can become too conservative for larger baskets.

### Selection Frequency

`historical_selection_frequency_local_search.png` shows that the local-search optimizer is mostly selecting defensive/risk-balancing ETFs in the historical-mean setup:

- `SHY`, `IEF`, and `HYG` dominate,
- `GLD` and `LQD` appear frequently,
- high-beta equity ETFs appear much less often.

This supports the interpretation that the QUBO optimizer is acting as a risk-aware portfolio selector, not just a return-ranking tool.

### Solver Scaling

`solver_runtime_scaling.png` and `solver_gap_scaling.png` summarize the synthetic Phase 5 scaling benchmark.

Presentation use:

- show exact full-QUBO enumeration growing quickly,
- show exact constrained enumeration as a correctness oracle rather than a scalable production solver,
- show local search as the strongest practical baseline,
- show annealing-style methods as feasible but noisier under this simple configuration.

## Slide Story

The best visual sequence is:

1. Penalty feasibility: "the QUBO formulation is correct only when the constraint penalty is calibrated."
2. Historical equity curves: "risk-aware sparse optimization improves smoothness and Sharpe in one regime."
3. Risk-return scatter: "top-K buys return with volatility; QUBO controls the tradeoff."
4. LASSO equity curves: "ML forecasts can change the winner; top-K can beat the optimizer if the risk objective is too conservative."
5. q/K Sharpe delta: "the method helps in specific regimes, especially moderate-cardinality portfolios with enough risk aversion."
6. Solver scaling: "the CS contribution includes runtime/optimality tradeoffs, not only finance metrics."
7. Selection frequency: "the optimizer's behavior is interpretable."

This gives the project a more credible CS-finance conclusion than claiming quantum-inspired optimization always wins.
