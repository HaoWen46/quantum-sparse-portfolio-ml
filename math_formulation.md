# Math Formulation

This file defines the first-pass model we should implement and present.

## Data And Forecasting Inputs

At rebalance date `t`, suppose we have `n` tradable assets.

- `r_{i,t+1}`: next-period return of asset `i`.
- `X_{i,t}`: features available at time `t`.
- `mu_i`: predicted next-period expected return from a course-covered ML model.
- `Sigma`: covariance matrix estimated from trailing returns.
- `x_i in {0,1}`: binary decision variable, where `x_i = 1` means select asset `i`.
- `K`: desired number of assets in the sparse portfolio.

The ML stage can estimate `mu_i` using:

- LASSO / elastic net regression,
- random forest regression or classification,
- gradient boosting regression or classification.

For a classification version, `mu_i` can be replaced by a ranking score such as `P(r_{i,t+1} > median cross-sectional return | X_{i,t})`.

## MVP Portfolio Interpretation

The simplest sparse portfolio is equal-weighted among selected assets:

`w_i = x_i / K`.

Then:

Expected return:

`R(x) = mu' x / K`.

Variance:

`V(x) = x' Sigma x / K^2`.

If `q_p` is the risk aversion for the equal-weight portfolio actually traded, the natural objective is:

`min_x q_p x' Sigma x / K^2 - mu' x / K`.

Multiplying by the positive constant `K` gives the equivalent binary-selection objective:

`min_x (q_p / K) x' Sigma x - mu' x`.

In code, user-facing experiment parameter `q` means portfolio-level risk aversion `q_p`; before building the QUBO, it is converted to:

`q_binary = q / K`.

## Constrained Binary Mean-Variance Problem

The clean constrained binary objective is:

`min_x q x' Sigma x - mu' x`

subject to:

`1' x = K`

and:

`x_i in {0,1}`.

Here `q > 0` is binary-selection risk aversion. In the backtests, this is `q_binary = q_p / K`, where `q_p` is the portfolio-level risk aversion shown in experiment tables. Higher `q` favors lower-risk baskets; lower `q` favors higher predicted-return baskets.

This exactly matches the common binary portfolio selection formulation used in Qiskit Finance, with `K` assets selected.

## QUBO Penalty Form

QUBO is unconstrained, so encode the cardinality constraint with a quadratic penalty:

`E(x) = q x' Sigma x - mu' x + A (1' x - K)^2`

where `A > 0` is a penalty coefficient.

For binary variables, `x_i^2 = x_i`, and:

`(1' x - K)^2 = (sum_i x_i - K)^2`

`= sum_i x_i + 2 sum_{i<j} x_i x_j - 2K sum_i x_i + K^2`

`= (1 - 2K) sum_i x_i + 2 sum_{i<j} x_i x_j + K^2`.

The constant `A K^2` does not affect the minimizer, but it may appear as a QUBO offset.

## Upper-Triangular QUBO Matrix

Using the convention:

`E(x) = sum_{i <= j} Q_{ij} x_i x_j + const`,

and assuming `Sigma` is symmetric:

Risk term:

`x' Sigma x = sum_i Sigma_{ii} x_i + 2 sum_{i<j} Sigma_{ij} x_i x_j`.

Therefore:

`Q_{ii} = q Sigma_{ii} - mu_i + A(1 - 2K)`

`Q_{ij} = 2q Sigma_{ij} + 2A`, for `i < j`.

The omitted constant is:

`const = A K^2`.

If a library uses a full matrix convention `x'Qx`, we must be careful not to double-count off-diagonal entries. For `dimod` and many QUBO dictionaries, the upper-triangular convention is the safest.

## Penalty Scaling

Penalty tuning is not cosmetic; it is a core experiment.

If `A` is too small, the solver may prefer portfolios with the wrong number of assets. If `A` is too large, the objective becomes dominated by feasibility and energy differences among feasible portfolios can become numerically small.

Practical starting rule:

`A = c * (q sum_{i,j} |Sigma_{ij}| + sum_i |mu_i|)`

with `c` in `{0.1, 0.3, 1, 3, 10}`.

Experiment:

- sweep `A`,
- record feasibility rate,
- objective gap among feasible samples,
- out-of-sample portfolio metrics.

For small `n`, exact enumeration can identify the smallest `A` that makes the QUBO minimizer feasible.

## Turnover Penalty

Let `x_prev` be the previous selected portfolio. Add turnover control:

`C sum_i (x_i - x_prev_i)^2`.

Since `x_i^2 = x_i` and `x_prev_i` is constant:

`(x_i - x_prev_i)^2 = x_i - 2 x_prev_i x_i + const`.

So turnover is just a linear adjustment:

`Q_{ii} += C (1 - 2 x_prev_i)`.

This is QUBO-friendly and useful for realistic backtesting.

## Sector Or Group Exposure Penalties

For a group `G`, if we want exactly `L_G` selected assets:

`A_G (sum_{i in G} x_i - L_G)^2`.

This expands exactly like the cardinality penalty but only within the group.

For soft upper or lower group bounds, exact QUBO encoding may require slack variables. For the MVP, use equality-style group targets or skip group constraints.

## Two-Stage Hybrid Extension

The equal-weight model is clean, but real portfolios often allow unequal weights. A practical extension is:

Stage 1: QUBO selects sparse asset set `S = {i: x_i = 1}`.

Stage 2: solve a continuous convex mean-variance problem over selected assets:

`min_w q w_S' Sigma_SS w_S - mu_S' w_S`

subject to:

`1' w_S = 1`,

`0 <= w_i <= u_i`.

This hybrid approach keeps the QUBO small and lets classical convex optimization handle continuous weights.

Presentation framing:

- QUBO solves combinatorial selection.
- Classical optimizer solves continuous allocation.
- This mirrors practical hybrid quantum-classical workflows.

## Validation Checks

For each rebalance date:

1. Confirm `Sigma` is symmetric.
2. Confirm `Sigma` is positive semi-definite or use shrinkage/diagonal loading.
3. For exact small instances, enumerate all `2^n` vectors and confirm:
   - constrained optimum among `sum x = K`,
   - QUBO optimum for each penalty `A`,
   - feasibility of the QUBO optimum.
4. Compare solver samples against exact optimum:
   - energy gap,
   - feasible objective gap,
   - cardinality violation.
5. In backtests, train forecasts using only past data.

## Main Failure Modes

- Expected return forecasts are noisy and dominate results.
- Naive random cross-validation leaks future information.
- Penalty `A` overwhelms the objective.
- Covariance estimates are unstable for too many assets and too little history.
- Solver energy is not the same thing as out-of-sample finance performance.

These are not bugs in the project. They are good analysis points.
