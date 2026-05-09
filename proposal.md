# Project Proposal: Quantum-Inspired Sparse Portfolio Optimization

Presentation date: June 9, 2026

## 1. Motivation

Portfolio construction is a natural setting where machine learning and optimization meet. Machine learning can produce forecasts or ranking signals for assets, but those predictions are not directly tradable until they are converted into a portfolio decision. In practice, investors also face constraints: they may want to hold only a small number of assets, limit concentration, reduce turnover, or control risk.

This project proposes a computer-science-oriented financial machine learning study. The idea is to use course-covered ML methods to estimate expected returns, then convert the sparse portfolio selection problem into a Quadratic Unconstrained Binary Optimization (QUBO) problem. This formulation is "quantum-inspired" because QUBO and Ising models are standard interfaces for quantum annealing and related optimization methods, while still being fully testable with classical solvers.

The goal is not to claim quantum advantage or to predict markets perfectly. The goal is to study whether a binary optimization layer can turn noisy ML forecasts into better constrained portfolio decisions than simple ranking rules.

## 2. Research Question

The central question is:

> Given machine-learned return forecasts and covariance-based risk estimates, can a QUBO-based sparse optimizer produce competitive constrained portfolios, and when does it improve on simple top-K ranking?

Subquestions include:

1. How should a cardinality-constrained portfolio problem be written as a QUBO?
2. How sensitive is the solution to the penalty parameter used to enforce exactly K selected assets?
3. Do risk-aware QUBO portfolios improve volatility, drawdown, or Sharpe ratio compared with selecting the top-K predicted-return assets?
4. How close can practical heuristic solvers, such as greedy search, local search, and simulated annealing, get to exact enumeration on small instances?
5. What does this imply for future quantum or hybrid quantum-classical optimization in finance?

## 3. Course Alignment

This project fits the course requirement because it uses machine learning algorithms and methods covered in ECON7225.

The forecasting layer can use:

- LASSO or elastic net regression,
- tree-based methods such as random forests or gradient boosting,
- cross-validation and model selection,
- walk-forward validation for financial time series.

The financial machine learning component is directly connected to the course's Financial Machine Learning topic. The optimization layer gives the project a stronger computer science angle while still depending on course-relevant ML forecasts and validation design.

## 4. Proposed Method

At each rebalance date, the project will estimate:

- `mu`: predicted asset returns or scores,
- `Sigma`: covariance matrix of asset returns,
- `K`: target number of assets to select.

The sparse portfolio decision is represented with binary variables:

```text
x_i = 1 if asset i is selected, and x_i = 0 otherwise.
```

The core optimization problem is:

```math
\min_x \; q x^\top \Sigma x - \mu^\top x
```

subject to:

```math
\sum_i x_i = K.
```

The constraint can be converted into a QUBO penalty:

```math
E(x) = q x^\top \Sigma x - \mu^\top x + A\left(\sum_i x_i - K\right)^2.
```

This makes the problem compatible with exact enumeration, local-search heuristics, simulated annealing, and future quantum-inspired or quantum-hybrid solvers.

## 5. Data And Experimental Design

The empirical part will use a liquid ETF universe, such as broad equity, bond, commodity, sector, and style ETFs. Daily price data will be converted into daily returns.

The study will use a walk-forward backtest:

1. train forecasting models using only past data,
2. estimate covariance from trailing returns,
3. solve the sparse portfolio selection problem,
4. hold the selected equal-weight portfolio until the next rebalance,
5. evaluate out-of-sample performance.

Candidate forecasting models:

- historical mean baseline,
- LASSO regression on lagged returns and volatility features,
- gradient boosting or random forest as a nonlinear robustness check.

Candidate solvers:

- exact enumeration for small instances,
- top-K by forecast as a simple baseline,
- greedy forward selection,
- swap-based local search,
- simulated annealing or D-Wave `neal` as a quantum-inspired baseline.

## 6. Expected Results

I do not expect the quantum-inspired optimizer to universally dominate simple baselines. Financial return forecasts are noisy, and a more complex optimizer cannot fix a weak signal by itself.

The expected result is more nuanced:

- Top-K ranking may achieve higher raw return when the forecast signal is strong and risk is not heavily penalized.
- QUBO-based selection may reduce volatility or drawdown by accounting for covariance among selected assets.
- Local search may perform close to exact enumeration while scaling better.
- Simulated annealing may be feasible but sensitive to penalty scaling and solver parameters.
- The benefit of QUBO may depend on the portfolio size K and risk-aversion parameter q.

The most useful final conclusion would be a regime-based statement, such as:

> QUBO sparse portfolio optimization is useful as a controllable decision layer after ML forecasting, especially when risk control and portfolio constraints matter, but it is not a universal replacement for simple ranking baselines.

## 7. Why The Quantum-Inspired Angle Matters

The quantum-inspired part matters because many realistic portfolio problems become combinatorial once discrete constraints are added. Examples include cardinality limits, turnover constraints, lot sizes, sector buckets, and multi-period trading decisions.

Current quantum hardware is not expected to beat classical solvers in this class project. However, QUBO is a useful modeling language because it is shared by:

- classical local-search algorithms,
- simulated annealing,
- quantum annealing,
- hybrid quantum-classical optimization workflows.

For economics and ML students, the broader lesson is that prediction is only one layer of the problem. Econometrics and ML estimate uncertain quantities such as returns and risks; optimization turns those estimates into actions. If quantum optimization becomes more useful in the future, economists will still need to formulate objectives carefully, estimate inputs responsibly, and validate decisions out of sample.

## 8. Planned Deliverables

The final project will include:

- a mathematical QUBO formulation of sparse portfolio selection,
- a tested implementation of the optimization pipeline,
- comparison of exact, heuristic, and annealing-style solvers,
- walk-forward backtests using course-covered ML forecasts,
- sensitivity analysis over K, risk aversion, and penalty strength,
- presentation figures showing risk-return tradeoffs and solver behavior.

## 9. Risks And Limitations

The main risks are:

- ML return forecasts may be weak out of sample,
- penalty tuning may strongly affect feasibility and solver quality,
- exact enumeration will not scale to large asset universes,
- simulated annealing may underperform simple local search without careful tuning.

These risks are part of the research design rather than failures. A credible result can show when QUBO helps, when it does not, and what must be true for quantum-inspired optimization to be useful in financial ML.

## 10. Working Title

Quantum-Inspired Sparse Portfolio Optimization with Machine-Learned Returns
