"""Small deterministic QUBO demo for Phase 1.

Run with:

    uv --cache-dir .uv-cache run python scripts/phase1_toy_qubo.py
"""

from __future__ import annotations

import numpy as np

from sparse_portfolio import (
    build_cardinality_qubo,
    dimod_exact_qubo,
    exact_cardinality_optimum,
    exact_qubo_optimum,
    greedy_forward_selection,
    local_search_swaps,
    neal_simulated_annealing_qubo,
    portfolio_objective,
    simulated_annealing_qubo,
    top_k_by_mu,
)


def main() -> None:
    tickers = np.array(["SPY", "QQQ", "TLT", "GLD", "VNQ"])
    mu = np.array([0.090, 0.080, 0.070, 0.020, 0.010])
    sigma = np.array(
        [
            [0.040, 0.030, -0.004, 0.002, 0.020],
            [0.030, 0.060, -0.006, 0.004, 0.025],
            [-0.004, -0.006, 0.020, 0.001, -0.002],
            [0.002, 0.004, 0.001, 0.025, 0.003],
            [0.020, 0.025, -0.002, 0.003, 0.050],
        ]
    )

    cardinality = 2
    risk_aversion = 0.25

    constrained = exact_cardinality_optimum(
        mu, sigma, cardinality=cardinality, risk_aversion=risk_aversion
    )
    selected = tickers[constrained.x.astype(bool)]
    print("Exact constrained optimum")
    print(f"  selected: {', '.join(selected)}")
    print(f"  objective: {constrained.value:.6f}")
    print()

    print("Penalty sweep")
    print("  penalty | selected        | card | feasible | base_objective | qubo_energy")
    print("  --------+-----------------+------+----------+----------------+------------")
    for penalty in [0.0, 0.001, 0.01, 0.05, 0.1, 1.0]:
        qubo, offset = build_cardinality_qubo(
            mu,
            sigma,
            cardinality=cardinality,
            risk_aversion=risk_aversion,
            penalty=penalty,
        )
        solution = exact_qubo_optimum(qubo, offset)
        names = ",".join(tickers[solution.x.astype(bool)])
        base_value = portfolio_objective(solution.x, mu, sigma, risk_aversion)
        feasible = solution.cardinality == cardinality
        print(
            f"  {penalty:7.3g} | {names:<15} |"
            f" {solution.cardinality:4d} | {str(feasible):<8} |"
            f" {base_value:14.6f} | {solution.value:10.6f}"
        )

    print()
    print("Classical baselines at penalty = 0.1")
    qubo, offset = build_cardinality_qubo(
        mu,
        sigma,
        cardinality=cardinality,
        risk_aversion=risk_aversion,
        penalty=0.1,
    )
    baselines = {
        "top_k_mu": top_k_by_mu(mu, sigma, cardinality, risk_aversion),
        "greedy": greedy_forward_selection(mu, sigma, cardinality, risk_aversion),
        "local_search": local_search_swaps(mu, sigma, cardinality, risk_aversion),
        "dimod_exact": dimod_exact_qubo(qubo, offset),
        "annealing_qubo": simulated_annealing_qubo(
            qubo,
            offset,
            seed=11,
            num_reads=64,
            num_steps=500,
            initial_temperature=2.0,
            final_temperature=1e-4,
        ),
        "neal_qubo": neal_simulated_annealing_qubo(
            qubo,
            offset,
            seed=11,
            num_reads=64,
            num_sweeps=500,
        ),
    }
    for name, solution in baselines.items():
        selected = ",".join(tickers[solution.x.astype(bool)])
        if name == "annealing_qubo":
            base_value = portfolio_objective(solution.x, mu, sigma, risk_aversion)
        else:
            base_value = solution.value
        print(f"  {name:<14} {selected:<12} base_objective={base_value:.6f}")


if __name__ == "__main__":
    main()
