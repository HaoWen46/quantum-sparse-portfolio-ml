import itertools

import numpy as np
import pytest

from sparse_portfolio import (
    build_cardinality_qubo,
    exact_cardinality_optimum,
    exact_qubo_optimum,
    penalized_objective,
    portfolio_objective,
    qubo_energy,
)


def test_build_cardinality_qubo_coefficients_match_formula():
    mu = np.array([0.10, 0.20, 0.05])
    sigma = np.array(
        [
            [0.30, 0.02, 0.01],
            [0.02, 0.40, 0.03],
            [0.01, 0.03, 0.20],
        ]
    )

    qubo, offset = build_cardinality_qubo(
        mu, sigma, cardinality=2, risk_aversion=0.5, penalty=3.0
    )

    assert offset == pytest.approx(12.0)
    assert qubo[(0, 0)] == pytest.approx(0.5 * 0.30 - 0.10 + 3.0 * (1 - 4))
    assert qubo[(1, 1)] == pytest.approx(0.5 * 0.40 - 0.20 + 3.0 * (1 - 4))
    assert qubo[(2, 2)] == pytest.approx(0.5 * 0.20 - 0.05 + 3.0 * (1 - 4))
    assert qubo[(0, 1)] == pytest.approx(2 * 0.5 * 0.02 + 2 * 3.0)
    assert qubo[(0, 2)] == pytest.approx(2 * 0.5 * 0.01 + 2 * 3.0)
    assert qubo[(1, 2)] == pytest.approx(2 * 0.5 * 0.03 + 2 * 3.0)


def test_qubo_energy_matches_direct_penalized_objective():
    mu = np.array([0.04, 0.07, -0.01, 0.03])
    sigma = np.array(
        [
            [0.10, 0.01, 0.02, 0.00],
            [0.01, 0.20, 0.03, 0.01],
            [0.02, 0.03, 0.15, 0.02],
            [0.00, 0.01, 0.02, 0.12],
        ]
    )
    cardinality = 2
    risk_aversion = 0.7
    penalty = 4.0

    qubo, offset = build_cardinality_qubo(
        mu, sigma, cardinality, risk_aversion, penalty
    )

    for bits in itertools.product((0, 1), repeat=4):
        expected = penalized_objective(
            bits, mu, sigma, cardinality, risk_aversion, penalty
        )
        actual = qubo_energy(bits, qubo, offset)
        assert actual == pytest.approx(expected)


def test_exact_cardinality_optimum_uses_constrained_objective():
    mu = np.array([0.30, 0.20, 0.01])
    sigma = np.eye(3)

    solution = exact_cardinality_optimum(
        mu, sigma, cardinality=2, risk_aversion=0.0
    )

    assert solution.x.tolist() == [1, 1, 0]
    assert solution.cardinality == 2
    assert solution.value == pytest.approx(portfolio_objective(solution.x, mu, sigma, 0.0))


def test_large_penalty_makes_qubo_optimum_feasible_and_consistent():
    mu = np.array([0.30, 0.20, 0.01, 0.00])
    sigma = np.eye(4) * 0.05
    cardinality = 2
    risk_aversion = 1.0
    penalty = 10.0

    constrained = exact_cardinality_optimum(
        mu, sigma, cardinality, risk_aversion
    )
    qubo, offset = build_cardinality_qubo(
        mu, sigma, cardinality, risk_aversion, penalty
    )
    unconstrained = exact_qubo_optimum(qubo, offset)

    assert unconstrained.cardinality == cardinality
    assert unconstrained.x.tolist() == constrained.x.tolist()
    assert portfolio_objective(unconstrained.x, mu, sigma, risk_aversion) == pytest.approx(
        constrained.value
    )


def test_input_validation_rejects_non_symmetric_sigma():
    mu = np.array([0.1, 0.2])
    sigma = np.array([[1.0, 0.2], [0.1, 1.0]])

    with pytest.raises(ValueError, match="symmetric"):
        build_cardinality_qubo(mu, sigma, cardinality=1, risk_aversion=1.0, penalty=1.0)
