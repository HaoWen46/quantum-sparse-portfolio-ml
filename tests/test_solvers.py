import numpy as np
import pytest

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


def test_top_k_by_mu_selects_largest_expected_returns():
    mu = np.array([0.01, 0.10, 0.04, 0.08])
    sigma = np.eye(4)

    solution = top_k_by_mu(mu, sigma, cardinality=2, risk_aversion=0.0)

    assert solution.x.tolist() == [0, 1, 0, 1]
    assert solution.value == portfolio_objective(solution.x, mu, sigma, 0.0)


def test_greedy_forward_selection_finds_simple_diagonal_case():
    mu = np.array([0.20, 0.19, 0.01])
    sigma = np.eye(3) * 0.01

    solution = greedy_forward_selection(mu, sigma, cardinality=2, risk_aversion=1.0)
    exact = exact_cardinality_optimum(mu, sigma, cardinality=2, risk_aversion=1.0)

    assert solution.x.tolist() == exact.x.tolist()
    assert solution.value == exact.value


def test_local_search_swaps_improves_bad_initial_solution():
    mu = np.array([0.20, 0.19, 0.01, 0.00])
    sigma = np.eye(4) * 0.01
    initial_x = np.array([0, 0, 1, 1])

    solution = local_search_swaps(
        mu,
        sigma,
        cardinality=2,
        risk_aversion=1.0,
        initial_x=initial_x,
    )
    exact = exact_cardinality_optimum(mu, sigma, cardinality=2, risk_aversion=1.0)

    assert solution.x.tolist() == exact.x.tolist()
    assert solution.value < portfolio_objective(initial_x, mu, sigma, 1.0)


def test_simulated_annealing_qubo_reaches_toy_exact_solution_with_seed():
    mu = np.array([0.30, 0.20, 0.01, 0.00])
    sigma = np.eye(4) * 0.05
    qubo, offset = build_cardinality_qubo(
        mu, sigma, cardinality=2, risk_aversion=1.0, penalty=10.0
    )

    exact = exact_qubo_optimum(qubo, offset)
    annealed = simulated_annealing_qubo(
        qubo,
        offset,
        seed=7,
        num_reads=64,
        num_steps=500,
        initial_temperature=2.0,
        final_temperature=1e-4,
    )

    assert annealed.value == pytest.approx(exact.value)
    assert annealed.x.tolist() == exact.x.tolist()


def test_dimod_exact_qubo_matches_internal_exact_solver():
    mu = np.array([0.30, 0.20, 0.01, 0.00])
    sigma = np.eye(4) * 0.05
    qubo, offset = build_cardinality_qubo(
        mu, sigma, cardinality=2, risk_aversion=1.0, penalty=10.0
    )

    internal = exact_qubo_optimum(qubo, offset)
    external = dimod_exact_qubo(qubo, offset)

    assert external.value == internal.value
    assert external.x.tolist() == internal.x.tolist()


def test_neal_simulated_annealing_qubo_reaches_toy_exact_solution_with_seed():
    mu = np.array([0.30, 0.20, 0.01, 0.00])
    sigma = np.eye(4) * 0.05
    qubo, offset = build_cardinality_qubo(
        mu, sigma, cardinality=2, risk_aversion=1.0, penalty=10.0
    )

    exact = exact_qubo_optimum(qubo, offset)
    annealed = neal_simulated_annealing_qubo(
        qubo,
        offset,
        seed=7,
        num_reads=64,
        num_sweeps=500,
    )

    assert annealed.value == pytest.approx(exact.value)
    assert annealed.x.tolist() == exact.x.tolist()
