# Presentation Outline

Working title:

> Quantum-Inspired Sparse Portfolio Optimization

Core claim:

> QUBO sparse optimization is a rigorous CS way to turn noisy financial ML forecasts into constrained portfolios, but its realized value depends on penalty scaling, risk aversion, cardinality, and solver quality.

## Slide 1: Project Question

Question:

Can ML forecasts plus a QUBO sparse optimizer produce competitive constrained portfolios, and when is the optimizer better than simple top-K ranking?

Positioning:

- financial ML supplies forecasts,
- QUBO supplies the CS/optimization artifact,
- backtesting checks whether the method survives out of sample.

## Slide 2: Pipeline

Visual:

`market data -> features -> ML forecast -> risk estimate -> QUBO -> solver -> monthly portfolio -> backtest`

Key message:

The project is a full experimental system, not just an econometrics exercise.

## Slide 3: Optimization Model

Formula:

`min_x q x' Sigma x - mu' x + A (1' x - K)^2`, with `x_i in {0,1}`.

Explain:

- `mu`: forecast return score,
- `Sigma`: estimated covariance,
- `K`: exact portfolio size,
- `q`: binary-selection risk aversion; in equal-weight backtests, `q_binary = q_portfolio / K`,
- `A`: cardinality penalty.

Key message:

The QUBO converts sparse selection into a binary energy minimization problem compatible with exact search, local search, and annealing-style samplers.

## Slide 4: Penalty Validation

Figure:

`results/figures/penalty_feasibility.png`

Key message:

The cardinality penalty is not cosmetic. Too small an `A` gives infeasible portfolios; around the tested `0.1` multiplier, feasibility becomes reliable in the synthetic sweep.

## Slide 5: Solver Stack

Compare:

- exact constrained enumeration,
- top-K by forecast,
- greedy/local search,
- simulated annealing,
- D-Wave `neal` sampler.

Key message:

Exact enumeration is a correctness oracle for small instances; local search is the strongest practical baseline; annealing is the quantum-inspired comparison point.

## Slide 6: Solver Scaling

Figures:

- `results/figures/solver_runtime_scaling.png`
- `results/figures/solver_gap_scaling.png`

Key message:

Exact full-QUBO enumeration scales poorly, exact constrained enumeration is useful as a small-instance oracle, and local search is the strongest practical baseline in this implementation. Annealing-style solvers are feasible but noisier without additional tuning.

## Slide 7: Why Quantum-Inspired Matters

Visual:

`forecasting problem -> constrained decision problem -> binary energy landscape`

Key message:

Quantum computing is not better here because it predicts markets. If it becomes useful, it will be because constrained portfolio construction becomes a huge binary search problem once we add cardinality, turnover, lot sizes, sector limits, and multi-period decisions.

Talk track:

- classical convex Markowitz is already well served by classical solvers,
- cardinality and trading constraints make the problem combinatorial,
- QUBO/Ising is the common interface used by quantum annealers and variational quantum algorithms,
- learning this formulation now teaches ECON/ML students how predictions become optimized decisions.

Good line:

> The quantum computer would not replace econometrics; it would sit after econometrics, searching a harder decision space built from estimated returns, risks, and constraints.

## Slide 8: Historical-Mean Backtest

Figure:

`results/figures/historical_equity_curves.png`

Key results:

- exact Sharpe: `0.8954`,
- local search Sharpe: `0.8842`,
- top-K Sharpe: `0.8601`,
- top-K has higher raw return but much larger drawdown.

Key message:

With historical-mean forecasts and `K = 5, q_portfolio = 10`, risk-aware sparse optimization improves the risk-return tradeoff.

## Slide 9: LASSO Counterexample

Figure:

`results/figures/lasso_equity_curves.png`

Key results:

- top-K Sharpe: `1.1320`,
- exact Sharpe: `0.7855`,
- local search Sharpe: `0.7439`.

Key message:

The optimizer is not magic. With LASSO forecasts, top-K wins realized Sharpe because the QUBO objective is too conservative for that forecast/risk setup.

## Slide 10: Gradient Boosting Robustness

Figures:

- `results/figures/gradient_boosting_equity_curves.png`
- `results/figures/gradient_boosting_risk_return_scatter.png`

Key message:

The nonlinear ML robustness check still favors top-K on realized Sharpe, but exact/local QUBO are much closer than under LASSO. Local search exactly matches exact enumeration in this run.

## Slide 11: Risk-Return View

Figures:

- `results/figures/historical_risk_return_scatter.png`
- `results/figures/lasso_risk_return_scatter.png`
- `results/figures/gradient_boosting_risk_return_scatter.png`

Key message:

Top-K often buys higher annual return with substantially higher annual volatility. The QUBO/local-search portfolios are useful when the goal is a controlled risk-return tradeoff rather than maximum raw growth.

## Slide 12: When Does QUBO Help?

Figure:

`results/figures/historical_qk_sharpe_delta.png`

Key message:

QUBO helps in a middle regime:

- useful at `K = 5` with enough risk aversion,
- useful at `K = 8, q = 10`,
- less useful when `K = 3`,
- sometimes too conservative when `K` is large and `q` is very high.

This is the strongest research conclusion.

## Slide 13: What Is The Optimizer Doing?

Figure:

`results/figures/historical_selection_frequency_local_search.png`

Key message:

The local-search optimizer repeatedly selects defensive and diversifying ETFs such as `SHY`, `IEF`, `HYG`, `GLD`, and `LQD`. This explains why it has smoother equity curves and lower drawdowns.

## Slide 14: Conclusion

Final takeaway:

QUBO sparse portfolio optimization is most valuable as a disciplined decision layer after ML forecasting, not as a universal performance booster. The CS contribution is the correct binary formulation, solver benchmarking, feasibility validation, and regime analysis.

Good final line:

> The interesting result is not "quantum-inspired wins"; it is that QUBO gives a controllable sparse optimizer whose success depends on identifiable modeling regimes.

Future-facing final note:

> If quantum optimization becomes relevant, the ECON skill will still be the same: estimate the inputs responsibly, define the objective honestly, and test whether the decision rule works out of sample.

## Backup Slides

Useful backup material:

- exact QUBO coefficient derivation from `math_formulation.md`,
- solver benchmark CSV from `results/phase1/solver_benchmark.csv`,
- full summary tables from `phase3_results.md`,
- bibliography from `references.bib`.
