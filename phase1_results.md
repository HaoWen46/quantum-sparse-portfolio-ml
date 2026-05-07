# Phase 1 Results

Phase 1 goal: prove the QUBO formulation and solver plumbing on deterministic and synthetic instances before using market data.

## Verification Status

Command:

`uv --cache-dir .uv-cache run pytest`

Result:

`15 passed`

Covered checks:

- QUBO coefficients match the derived formula.
- QUBO energy equals the direct penalized objective.
- exact constrained enumeration works.
- exact full-QUBO enumeration works.
- large enough penalty recovers the constrained optimum.
- `dimod.ExactSolver` agrees with the internal exact solver.
- in-house simulated annealing and `neal` recover the toy optimum under fixed seeds.
- synthetic covariance matrices are positive semidefinite.

## Toy Demo

Command:

`uv --cache-dir .uv-cache run python scripts/phase1_toy_qubo.py`

The deterministic toy demo shows the intended penalty behavior:

- with `A = 0`, the QUBO over-selects 4 assets;
- with `A = 0.01` or `A = 0.05`, it still over-selects 3 assets;
- with `A = 0.1`, it recovers the exact `K = 2` feasible solution.

This is presentation-useful because it makes penalty tuning concrete rather than abstract.

## Random Synthetic Sweep

Command:

`uv --cache-dir .uv-cache run python scripts/phase1_random_sweep.py`

Generated files:

- `results/phase1/penalty_sweep.csv`
- `results/phase1/penalty_sweep_summary.csv`
- `results/phase1/solver_benchmark.csv`

Current summary from 25 synthetic `n = 10`, `K = 3` instances:

| Penalty Multiplier | Feasibility Rate | Mean Cardinality Violation |
| ---: | ---: | ---: |
| 0.000 | 0.000 | 5.960 |
| 0.001 | 0.000 | 5.360 |
| 0.010 | 0.000 | 2.000 |
| 0.030 | 0.000 | 1.000 |
| 0.100 | 1.000 | 0.000 |
| 0.300 | 1.000 | 0.000 |
| 1.000 | 1.000 | 0.000 |
| 3.000 | 1.000 | 0.000 |

Interpretation:

- penalty scaling is not cosmetic;
- below `0.1 * objective_scale`, QUBO optima are systematically infeasible in this synthetic setting;
- at or above `0.1 * objective_scale`, exact-QUBO optima match the exact constrained optima on this sweep.

## Solver Benchmark Snapshot

For one synthetic instance, the current solver table shows:

- exact constrained, exact QUBO, `dimod.ExactSolver`, greedy, local search, in-house annealing, and `neal` all reach the same optimum;
- top-K by expected return is feasible but worse, which gives us a useful baseline gap.

This is exactly the baseline pattern we want before moving to real data.

## Decision

Use both:

- internal exact/heuristic solvers for transparency and control;
- `dimod`/`neal` for recognized QUBO tooling.

Do not add heavier Qiskit dependencies unless the presentation specifically needs a QAOA/VQE demo, which is currently out of scope.

## Next Step

Move to Phase 2:

1. add/cached ETF price data workflow;
2. compute returns and features;
3. estimate `mu` with simple baselines first;
4. add LASSO/elastic net, random forest, and gradient boosting after the data pipeline is stable.
