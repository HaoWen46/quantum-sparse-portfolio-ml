# Research Brief

This project sits at the intersection of three literatures:

1. machine-learning forecasts for portfolio inputs,
2. sparse/cardinality-constrained portfolio optimization,
3. QUBO / Ising / annealing formulations.

## Core Papers And Takeaways

### Markowitz Portfolio Optimization

Markowitz mean-variance optimization is the base model: maximize expected return while controlling covariance-based risk. In continuous unconstrained or convex-constrained form, this is usually a quadratic program. The interesting computational difficulty appears when we add discrete constraints such as "select exactly K assets", minimum lots, cardinality limits, or transaction decisions.

Project use:
- We use Markowitz as the financial objective.
- We intentionally add sparsity/cardinality to make the problem combinatorial and QUBO-friendly.

### Qiskit Finance Portfolio Optimization Tutorial

URL: https://qiskit-community.github.io/qiskit-finance/tutorials/01_portfolio_optimization.html

Qiskit Finance presents the clean binary portfolio-selection objective:

`min_{x in {0,1}^n} q x' Sigma x - mu' x`

subject to:

`1' x = B`

where `x_i = 1` means asset `i` is selected, `mu` is expected return, `Sigma` is covariance, `q` is risk appetite, and `B` is the number of selected assets. The equality constraint is converted into a quadratic penalty `(1' x - B)^2`.

Project use:
- This is the MVP formulation.
- It is mathematically simple, course-compatible, and easy to validate by exact enumeration for small `n`.

### Sakuler et al. 2025: Real-World Quantum Annealing Test

URL: https://link.springer.com/article/10.1007/s42484-025-00268-2

This open-access paper is valuable because it is sober and practical. It formulates a real bank portfolio problem as QUBO, compares D-Wave hybrid solvers against a classical strategy, and emphasizes that QUBO penalty tuning is crucial. It also discusses realistic constraints: normalized weights, target volatility, lower/upper weight bounds, and grouped asset-class constraints.

Project use:
- Cite as the strongest "this is a real thing people test" source.
- Borrow the caution that quantum/hybrid methods should be benchmarked against exact or strong classical baselines.
- Borrow the idea that penalty coefficients are part of the research question.

### Lu et al. 2024: Quantum-Inspired Portfolio Optimization In The QUBO Framework

URL: https://ideas.repec.org/p/arx/papers/2410.05932.html

This arXiv paper explicitly uses the phrase "quantum-inspired" and focuses on QUBO portfolio optimization, penalty coefficient estimation, and preprocessing/two-stage search on real quarterly financial data.

Project use:
- Good source for our exact framing.
- Supports a "quantum-inspired without hardware" angle.
- Reinforces studying penalty scaling and preprocessing as first-class engineering choices.

### Palmer et al. 2021: Investment Bands And Target Volatility

URL: https://ideas.repec.org/p/arx/papers/2106.06735.html

This paper shows how to implement practical constraints such as target risk and individual investment bands as QUBO-compatible constraints for quantum optimization.

Project use:
- Useful if we extend beyond equal-weight selection.
- Provides motivation for constrained portfolios that avoid corner solutions.

### Palmer et al. 2022: Index Tracking With Cardinality Constraints

URL: https://ideas.repec.org/p/arx/papers/2208.11380.html

This paper focuses on cardinality constraints in quantum portfolio optimization, applying the method to enhanced index tracking with smaller portfolios.

Project use:
- Directly relevant to "sparse" portfolio construction.
- A useful fallback framing: instead of maximizing predicted return, track an index using a sparse subset.

### Buonaiuto et al. 2023: Best Practices On Real Quantum Devices

URL: https://www.nature.com/articles/s41598-023-45392-w

This Scientific Reports paper uses small Yahoo Finance examples, maps constrained integer quadratic portfolio optimization into QUBO, then studies VQE and hardware/hyperparameter choices. It is useful mainly as a best-practices warning: small instances, penalty choice, ansatz/optimizer choices, and simulator-vs-hardware gaps matter.

Project use:
- We should not promise quantum advantage.
- We can mention that our annealing benchmarks are intentionally small and diagnostic.

### Egger et al. 2020: Quantum Computing For Finance

URL: https://research.ibm.com/publications/quantum-computing-for-finance-state-of-the-art-and-future-prospects

This survey identifies simulation, optimization, and machine learning as the major finance problem classes where quantum computing may eventually be relevant. It is useful because it keeps the quantum-finance motivation broader than portfolio selection alone while still treating optimization as a central application area.

Project use:
- Supports the "future potential" section.
- Helps explain why this project belongs at the intersection of financial ML and computational optimization.
- Provides a sober framing: current technical challenges remain, but finance has many computationally difficult decision problems.

### IBM Quantum Portfolio Optimizer Documentation

URL: https://quantum.cloud.ibm.com/docs/guides/global-data-quantum-optimizer

IBM documents an experimental Qiskit Function for dynamic portfolio optimization. The workflow maps input price data and investment constraints into a QUBO, transforms it into an Ising Hamiltonian, and solves it using a VQE-style quantum workflow with post-processing. The documentation explicitly marks the function as preview-stage, which is perfect for our narrative: the direction is real, but it is not mature enough to justify hype.

Project use:
- Shows that QUBO portfolio optimization is becoming an actual software interface, not only a paper idea.
- Supports the claim that learning QUBO is useful preparation for future quantum/hybrid optimization tools.
- Reinforces the need to keep our own project solver-agnostic and benchmarked against classical baselines.

### Quinton et al. 2025: Quantum Annealing Challenges And Limitations

URL: https://www.nature.com/articles/s41598-025-96220-2

This paper is useful as a cautionary source on quantum annealing compared with classical solvers. The main project relevance is not a specific finance result, but the methodological warning: quantum annealing experiments need careful problem embedding, parameter choices, and classical comparisons before any advantage claim is credible.

Project use:
- Cite as a reason to avoid "quantum is better today" language.
- Motivates our exact/local-search baselines and objective-gap reporting.
- Useful backup source if the presentation is challenged on whether annealing is actually competitive.

### D-Wave Portfolio Optimization Example

URL: https://github.com/dwave-examples/portfolio-optimization

D-Wave maintains a portfolio-optimization example that models stock purchases with risk, returns, budget, transaction costs, and several CQM/DQM formulations. The README is useful because it distinguishes three objective styles:

- bi-objective mean-variance,
- maximize return subject to risk bound,
- minimize risk subject to return bound.

It also includes a multi-period rebalancing demo and explicitly grid-searches risk-aversion and penalty coefficients for some formulations.

Project use:
- Good engineering reference, especially for CLI/demo shape.
- Confirms that penalty/risk-aversion grid search is a normal part of this problem.
- We should not copy its CQM-first formulation for the MVP; our class project is cleaner as QUBO-first binary selection.

### Lucas 2014: Ising Formulations Of Many NP Problems

URL: https://www.frontiersin.org/journals/physics/articles/10.3389/fphy.2014.00005/full

Lucas is a standard reference for mapping combinatorial optimization problems into Ising Hamiltonians. It is not finance-specific, but it gives theoretical legitimacy to the QUBO/Ising formulation style.

Project use:
- Cite for the general QUBO/Ising mapping idea.
- Helpful for explaining why binary optimization and annealing belong together.

### ML Forecasts For Portfolio Inputs

Useful sources:

- Machine learning-based portfolio optimization, Financial Innovation 2026:
  https://link.springer.com/article/10.1186/s40854-026-00927-8
- Portfolio optimization with return prediction using deep learning and machine learning, Expert Systems with Applications 2021:
  https://www.sciencedirect.com/science/article/pii/S0957417420307521
- Machine Learning Portfolio Allocation, Journal of Finance and Data Science 2022:
  https://www.sciencedirect.com/science/article/pii/S2405918821000155
- Automated stock picking using random forests, Journal of Empirical Finance 2023:
  https://www.sciencedirect.com/science/article/pii/S0927539823000452

Project use:
- We only need modest ML forecasting. The optimizer benchmark is the main story.
- Use LASSO/elastic net, random forests, and gradient boosting because they are course-covered.
- Avoid deep learning unless we need a stretch comparison.

## Library Survey

### D-Wave Ocean: `dimod`

URL: https://docs.dwavequantum.com/en/latest/ocean/api_ref_dimod/models.html

`dimod` provides Binary Quadratic Models (BQMs), which encode QUBO or Ising models. The BQM energy form is:

`E(v) = sum_i a_i v_i + sum_{i<j} b_ij v_i v_j + c`

with variables in either `{0,1}` or `{-1,+1}`.

Use:
- Strong candidate for representing QUBO cleanly.
- Also has exact solver utilities for tiny instances.

### D-Wave `neal`

URL: https://dwave-neal-docs.readthedocs.io/en/latest/reference/sampler.html

`neal` is a simulated annealing sampler compatible with `dimod` BQMs. It supports sampling BQM, Ising, and QUBO objectives.

Use:
- Good MVP annealing backend.
- Does not require quantum hardware.

### `dimod.ExactSolver`

URL: https://docs.dwavequantum.com/en/latest/ocean/api_ref_dimod/generated/dimod.reference.samplers.ExactSolver.sample.html

`dimod.ExactSolver` returns all possible solutions for a BQM. This is only feasible for small `n`, but perfect for validating the QUBO expansion and penalty behavior.

Use:
- Phase 1 exact verification for tiny generated instances.
- Cross-check our own enumeration implementation.

### OpenJij

URL: https://www.openjij.org/

OpenJij is an open-source Python interface for fast simulated annealing and simulated quantum annealing for QUBO/Ising models.

Use:
- Good "quantum-inspired" solver candidate.
- Could compare against `neal` if installation works cleanly.

### PyQUBO

URL: https://pyqubo.readthedocs.io/en/latest/getting_started.html

PyQUBO lets us define Hamiltonians symbolically and compile them into QUBO, Ising, or `dimod.BinaryQuadraticModel`. It can reduce higher-order polynomial expressions into quadratic form with auxiliary variables.

Use:
- Useful if constraints get complex.
- Not necessary for the first MVP because the objective is already quadratic.

### Qiskit Finance

URL: https://qiskit-community.github.io/qiskit-finance/tutorials/01_portfolio_optimization.html

Qiskit Finance gives a clean reference implementation of binary portfolio optimization and supports quantum algorithms such as QAOA/SamplingVQE. However, the package ecosystem can be heavier and more fragile than D-Wave Ocean for a class project.

Use:
- Use as formula reference.
- Avoid making it a core dependency unless the environment cooperates.

### scikit-learn Ledoit-Wolf Covariance

URL: https://scikit-learn.org/stable/modules/generated/sklearn.covariance.LedoitWolf.html

`sklearn.covariance.LedoitWolf` estimates a regularized covariance matrix using an automatically computed shrinkage coefficient. scikit-learn documents the regularized covariance as:

`(1 - shrinkage) * cov + shrinkage * mu * I`

where `mu = trace(cov) / n_features`.

Use:
- Stabilize `Sigma` in the portfolio objective.
- This is especially useful when the asset universe is moderately large relative to the trailing return window.

### yfinance

URL: https://pypi.org/project/yfinance/

`yfinance` is a convenient open-source way to fetch Yahoo Finance market data for research and educational purposes. Its own package page emphasizes that it is not affiliated with Yahoo and that Yahoo Finance data is intended for personal use.

Use:
- Candidate data source for daily adjusted prices.
- Cache downloaded data locally once fetched.
- Cite/use only as educational project data, not production-grade market data.

## Research Positioning

Good title:

> Quantum-Inspired Sparse Portfolio Optimization With Machine-Learned Returns

Thesis:

> We combine course-covered ML predictors with a QUBO formulation of sparse portfolio selection, then study whether annealing-style solvers can match exact or greedy classical baselines under realistic validation, penalty scaling, and rebalance constraints.

What we should not claim:
- No quantum advantage claim.
- No "we predict markets" claim.
- No causal interpretation.
- No claim that quantum hardware is already better than classical solvers for this ETF experiment.

What we can claim:
- Correct QUBO formulation.
- Careful validation and leakage avoidance.
- Solver benchmarking and sensitivity analysis.
- A useful computational pipeline for sparse financial ML portfolios.
- A forward-looking interface: the same QUBO problem can be passed to classical heuristics today and quantum/hybrid solvers later.

## Why Quantum Might Matter Later

The honest answer is: quantum computing is not currently better for this class project. Classical local search is already excellent on the current 20-ETF setup. The future argument is about scale and constraint complexity.

Portfolio construction becomes much harder when the decision is not only "choose weights" but also:

- choose a sparse subset,
- limit the number of trades,
- satisfy sector, asset-class, tax, and liquidity rules,
- model discrete lots and minimum positions,
- rebalance over multiple future dates.

Those choices create a large binary decision space. QUBO/Ising models are important because they translate that space into an energy minimization problem, which is the natural input form for quantum annealing and a common intermediate form for gate-model variational algorithms.

Meaning for ECON/ML students:

- Econometrics remains essential because the inputs are estimated, noisy, and validation-sensitive.
- ML remains essential because return/risk signals may come from flexible predictive models.
- Optimization becomes the bridge from predictions to decisions.
- Quantum computing, if it becomes useful, would expand the decision layer; it would not eliminate the need for good statistical modeling.

## Phase 0 Decisions

Current locked choices:

- MVP objective: equal-weight binary subset selection with exact `K` assets.
- QUBO constraint: cardinality penalty `(1' x - K)^2`.
- Risk estimate: trailing covariance, preferably Ledoit-Wolf shrinkage.
- Forecasting models: start with historical mean, LASSO/elastic net, random forest, and gradient boosting.
- Solver stack: first own enumeration/greedy/local-search; then `dimod` + `neal`; then OpenJij if easy.
- Data: daily ETF/equity prices via cached CSV, likely from `yfinance`.
- Execution: all Python/package commands through `uv`.
