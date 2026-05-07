"""Benchmark solver runtime and objective gaps on synthetic QUBO instances."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from sparse_portfolio import (
    build_cardinality_qubo,
    exact_cardinality_optimum,
    exact_qubo_optimum,
    generate_synthetic_instance,
    greedy_forward_selection,
    local_search_swaps,
    neal_simulated_annealing_qubo,
    penalty_scale,
    portfolio_objective,
    simulated_annealing_qubo,
    top_k_by_mu,
)


def timed_call(function, *args, **kwargs):
    started = perf_counter()
    result = function(*args, **kwargs)
    return result, perf_counter() - started


def main() -> None:
    output_dir = Path("results/phase5")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | int | bool | str]] = []
    n_values = [8, 10, 12, 14, 16, 18, 20]
    seeds = [2026, 2027, 2028]
    cardinality = 5
    risk_aversion = 0.5
    penalty_multiplier = 1.0

    for n_assets in n_values:
        for seed in seeds:
            instance = generate_synthetic_instance(
                n_assets=n_assets,
                cardinality=cardinality,
                risk_aversion=risk_aversion,
                seed=seed,
            )
            exact, exact_seconds = timed_call(
                exact_cardinality_optimum,
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

            solvers = [
                (
                    "exact_constrained",
                    exact,
                    exact_seconds,
                ),
                (
                    "top_k_mu",
                    *timed_call(
                        top_k_by_mu,
                        instance.mu,
                        instance.sigma,
                        instance.cardinality,
                        instance.risk_aversion,
                    ),
                ),
                (
                    "greedy",
                    *timed_call(
                        greedy_forward_selection,
                        instance.mu,
                        instance.sigma,
                        instance.cardinality,
                        instance.risk_aversion,
                    ),
                ),
                (
                    "local_search",
                    *timed_call(
                        local_search_swaps,
                        instance.mu,
                        instance.sigma,
                        instance.cardinality,
                        instance.risk_aversion,
                    ),
                ),
                (
                    "annealing_qubo",
                    *timed_call(
                        simulated_annealing_qubo,
                        qubo,
                        offset,
                        num_reads=32,
                        num_steps=500,
                        seed=seed,
                        initial_temperature=2.0,
                        final_temperature=1e-4,
                    ),
                ),
                (
                    "neal_qubo",
                    *timed_call(
                        neal_simulated_annealing_qubo,
                        qubo,
                        offset,
                        num_reads=32,
                        num_sweeps=500,
                        seed=seed,
                        beta_range=(0.1, 100.0),
                    ),
                ),
            ]

            if n_assets <= 16:
                solvers.append(
                    (
                        "exact_qubo",
                        *timed_call(exact_qubo_optimum, qubo, offset),
                    )
                )

            for solver_name, solution, seconds in solvers:
                base_objective = portfolio_objective(
                    solution.x,
                    instance.mu,
                    instance.sigma,
                    instance.risk_aversion,
                )
                rows.append(
                    {
                        "n_assets": n_assets,
                        "cardinality": cardinality,
                        "seed": seed,
                        "solver": solver_name,
                        "runtime_seconds": seconds,
                        "selected_cardinality": int(solution.x.sum()),
                        "feasible": int(solution.x.sum()) == cardinality,
                        "base_objective": float(base_objective),
                        "gap_vs_exact": float(base_objective - exact.value),
                    }
                )
            print(f"Finished n={n_assets}, seed={seed}")

    results = pd.DataFrame(rows)
    output_path = output_dir / "solver_scaling.csv"
    results.to_csv(output_path, index=False)

    summary = (
        results.groupby(["n_assets", "solver"], as_index=False)
        .agg(
            mean_runtime_seconds=("runtime_seconds", "mean"),
            median_runtime_seconds=("runtime_seconds", "median"),
            mean_gap_vs_exact=("gap_vs_exact", "mean"),
            feasibility_rate=("feasible", "mean"),
        )
        .sort_values(["n_assets", "mean_runtime_seconds"])
    )
    summary_path = output_dir / "solver_scaling_summary.csv"
    summary.to_csv(summary_path, index=False)

    print()
    print("Solver scaling summary")
    print(summary.to_string(index=False, float_format="{:.6f}".format))
    print()
    print(f"Wrote {output_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
