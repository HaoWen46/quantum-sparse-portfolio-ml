"""Experiment helpers for Phase 1 QUBO validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from sparse_portfolio.qubo import (
    BinarySolution,
    build_cardinality_qubo,
    exact_cardinality_optimum,
    exact_qubo_optimum,
    portfolio_objective,
)
from sparse_portfolio.solvers import (
    dimod_exact_qubo,
    greedy_forward_selection,
    local_search_swaps,
    neal_simulated_annealing_qubo,
    simulated_annealing_qubo,
    top_k_by_mu,
)


@dataclass(frozen=True)
class SyntheticPortfolioInstance:
    """Synthetic portfolio inputs with a PSD covariance matrix."""

    mu: np.ndarray
    sigma: np.ndarray
    cardinality: int
    risk_aversion: float
    seed: int

    @property
    def n_assets(self) -> int:
        return len(self.mu)


def generate_synthetic_instance(
    *,
    n_assets: int = 10,
    cardinality: int = 3,
    risk_aversion: float = 0.5,
    seed: int = 0,
    n_factors: int = 3,
    expected_return_scale: float = 0.08,
) -> SyntheticPortfolioInstance:
    """Generate a small random mean-variance instance.

    The covariance matrix is built from a factor model plus positive
    idiosyncratic variance, so it is positive semidefinite by construction.
    """

    if n_assets <= 0:
        raise ValueError("n_assets must be positive")
    if not 0 <= cardinality <= n_assets:
        raise ValueError("cardinality must be between 0 and n_assets")
    if n_factors <= 0:
        raise ValueError("n_factors must be positive")
    if risk_aversion < 0:
        raise ValueError("risk_aversion must be non-negative")
    if expected_return_scale <= 0:
        raise ValueError("expected_return_scale must be positive")

    rng = np.random.default_rng(seed)
    loadings = rng.normal(0.0, 0.08, size=(n_assets, n_factors))
    factor_cov = loadings @ loadings.T
    idiosyncratic = rng.uniform(0.01, 0.06, size=n_assets)
    sigma = factor_cov + np.diag(idiosyncratic)
    mu = rng.normal(expected_return_scale, expected_return_scale / 2, size=n_assets)

    return SyntheticPortfolioInstance(
        mu=mu,
        sigma=sigma,
        cardinality=cardinality,
        risk_aversion=risk_aversion,
        seed=seed,
    )


def penalty_scale(
    mu: Iterable[float] | np.ndarray,
    sigma: Iterable[Iterable[float]] | np.ndarray,
    risk_aversion: float,
) -> float:
    """Return a conservative objective-scale proxy for penalty sweeps."""

    mu_arr = np.asarray(mu, dtype=float)
    sigma_arr = np.asarray(sigma, dtype=float)
    scale = risk_aversion * np.abs(sigma_arr).sum() + np.abs(mu_arr).sum()
    return float(max(scale, 1e-12))


def run_penalty_sweep(
    *,
    n_instances: int = 20,
    n_assets: int = 10,
    cardinality: int = 3,
    risk_aversion: float = 0.5,
    penalty_multipliers: Iterable[float] = (0.0, 0.001, 0.01, 0.1, 1.0, 10.0),
    seed: int = 0,
) -> pd.DataFrame:
    """Run exact QUBO penalty sweeps on synthetic instances."""

    rows: list[dict[str, float | int | bool]] = []
    multipliers = list(penalty_multipliers)

    for instance_index in range(n_instances):
        instance_seed = seed + instance_index
        instance = generate_synthetic_instance(
            n_assets=n_assets,
            cardinality=cardinality,
            risk_aversion=risk_aversion,
            seed=instance_seed,
        )
        constrained = exact_cardinality_optimum(
            instance.mu,
            instance.sigma,
            instance.cardinality,
            instance.risk_aversion,
        )
        base_scale = penalty_scale(instance.mu, instance.sigma, instance.risk_aversion)

        for multiplier in multipliers:
            penalty = multiplier * base_scale
            qubo, offset = build_cardinality_qubo(
                instance.mu,
                instance.sigma,
                instance.cardinality,
                instance.risk_aversion,
                penalty,
            )
            qubo_solution = exact_qubo_optimum(qubo, offset)
            base_value = portfolio_objective(
                qubo_solution.x,
                instance.mu,
                instance.sigma,
                instance.risk_aversion,
            )
            feasible = qubo_solution.cardinality == instance.cardinality
            feasible_gap = base_value - constrained.value if feasible else np.nan

            rows.append(
                {
                    "instance": instance_index,
                    "seed": instance_seed,
                    "n_assets": n_assets,
                    "cardinality": cardinality,
                    "risk_aversion": risk_aversion,
                    "penalty_multiplier": float(multiplier),
                    "penalty": float(penalty),
                    "selected_cardinality": qubo_solution.cardinality,
                    "cardinality_violation": abs(
                        qubo_solution.cardinality - instance.cardinality
                    ),
                    "feasible": feasible,
                    "base_objective": float(base_value),
                    "exact_constrained_objective": float(constrained.value),
                    "feasible_gap": float(feasible_gap),
                    "qubo_energy": float(qubo_solution.value),
                }
            )

    return pd.DataFrame(rows)


def summarize_penalty_sweep(sweep: pd.DataFrame) -> pd.DataFrame:
    """Aggregate penalty sweep output by multiplier."""

    return (
        sweep.groupby("penalty_multiplier", as_index=False)
        .agg(
            feasibility_rate=("feasible", "mean"),
            mean_cardinality_violation=("cardinality_violation", "mean"),
            mean_feasible_gap=("feasible_gap", "mean"),
            median_feasible_gap=("feasible_gap", "median"),
        )
        .sort_values("penalty_multiplier")
    )


def benchmark_solvers(
    instance: SyntheticPortfolioInstance,
    *,
    penalty_multiplier: float = 1.0,
    annealing_reads: int = 64,
    annealing_steps: int = 1_000,
    seed: int = 0,
) -> pd.DataFrame:
    """Compare exact and heuristic solvers on one synthetic instance."""

    exact = exact_cardinality_optimum(
        instance.mu,
        instance.sigma,
        instance.cardinality,
        instance.risk_aversion,
    )
    penalty = penalty_multiplier * penalty_scale(
        instance.mu,
        instance.sigma,
        instance.risk_aversion,
    )
    qubo, offset = build_cardinality_qubo(
        instance.mu,
        instance.sigma,
        instance.cardinality,
        instance.risk_aversion,
        penalty,
    )
    exact_qubo = exact_qubo_optimum(qubo, offset)

    solutions: dict[str, BinarySolution] = {
        "exact_constrained": exact,
        "exact_qubo": exact_qubo,
        "dimod_exact_qubo": dimod_exact_qubo(qubo, offset),
        "top_k_mu": top_k_by_mu(
            instance.mu,
            instance.sigma,
            instance.cardinality,
            instance.risk_aversion,
        ),
        "greedy": greedy_forward_selection(
            instance.mu,
            instance.sigma,
            instance.cardinality,
            instance.risk_aversion,
        ),
        "local_search": local_search_swaps(
            instance.mu,
            instance.sigma,
            instance.cardinality,
            instance.risk_aversion,
        ),
        "annealing_qubo": simulated_annealing_qubo(
            qubo,
            offset,
            seed=seed,
            num_reads=annealing_reads,
            num_steps=annealing_steps,
            initial_temperature=2.0,
            final_temperature=1e-4,
        ),
        "neal_qubo": neal_simulated_annealing_qubo(
            qubo,
            offset,
            seed=seed,
            num_reads=annealing_reads,
            num_sweeps=annealing_steps,
        ),
    }

    rows: list[dict[str, float | int | bool | str]] = []
    for name, solution in solutions.items():
        base_value = portfolio_objective(
            solution.x,
            instance.mu,
            instance.sigma,
            instance.risk_aversion,
        )
        rows.append(
            {
                "solver": name,
                "cardinality": solution.cardinality,
                "feasible": solution.cardinality == instance.cardinality,
                "base_objective": float(base_value),
                "gap_vs_exact": float(base_value - exact.value),
                "qubo_energy": float(solution.value)
                if name in {
                    "exact_qubo",
                    "dimod_exact_qubo",
                    "annealing_qubo",
                    "neal_qubo",
                }
                else np.nan,
            }
        )

    return pd.DataFrame(rows)
