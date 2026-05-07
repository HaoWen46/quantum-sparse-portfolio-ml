"""Run synthetic Phase 1 penalty and solver sweeps.

Run with:

    uv --cache-dir .uv-cache run python scripts/phase1_random_sweep.py
"""

from __future__ import annotations

from pathlib import Path

from sparse_portfolio import (
    benchmark_solvers,
    generate_synthetic_instance,
    run_penalty_sweep,
    summarize_penalty_sweep,
)


def main() -> None:
    output_dir = Path("results/phase1")
    output_dir.mkdir(parents=True, exist_ok=True)

    sweep = run_penalty_sweep(
        n_instances=25,
        n_assets=10,
        cardinality=3,
        risk_aversion=0.5,
        penalty_multipliers=(0.0, 0.001, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0),
        seed=2026,
    )
    summary = summarize_penalty_sweep(sweep)

    sweep_path = output_dir / "penalty_sweep.csv"
    summary_path = output_dir / "penalty_sweep_summary.csv"
    sweep.to_csv(sweep_path, index=False)
    summary.to_csv(summary_path, index=False)

    instance = generate_synthetic_instance(
        n_assets=10,
        cardinality=3,
        risk_aversion=0.5,
        seed=2026,
    )
    solvers = benchmark_solvers(
        instance,
        penalty_multiplier=1.0,
        annealing_reads=64,
        annealing_steps=1_000,
        seed=2026,
    )
    solvers_path = output_dir / "solver_benchmark.csv"
    solvers.to_csv(solvers_path, index=False)

    print("Penalty sweep summary")
    print(summary.to_string(index=False, float_format="{:.6f}".format))
    print()
    print("Solver benchmark")
    print(solvers.to_string(index=False, float_format="{:.6f}".format))
    print()
    print(f"Wrote {sweep_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {solvers_path}")


if __name__ == "__main__":
    main()
