"""QUBO construction and exact verification for sparse portfolios.

The MVP model is

    min_x q x' Sigma x - mu' x + A (1' x - K)^2,

where x is binary and K is the target cardinality.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from typing import Iterable

import numpy as np


Qubo = dict[tuple[int, int], float]


@dataclass(frozen=True)
class BinarySolution:
    """A binary vector and its objective/energy value."""

    x: np.ndarray
    value: float

    @property
    def cardinality(self) -> int:
        return int(self.x.sum())


def _as_binary_vector(x: Iterable[int] | np.ndarray, n: int | None = None) -> np.ndarray:
    vector = np.asarray(x, dtype=int)
    if vector.ndim != 1:
        raise ValueError("x must be a one-dimensional vector")
    if n is not None and vector.shape[0] != n:
        raise ValueError(f"x has length {vector.shape[0]}, expected {n}")
    if not np.all((vector == 0) | (vector == 1)):
        raise ValueError("x must contain only 0/1 entries")
    return vector


def _validate_inputs(
    mu: Iterable[float] | np.ndarray,
    sigma: Iterable[Iterable[float]] | np.ndarray,
    cardinality: int,
    risk_aversion: float,
    penalty: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    mu_arr = np.asarray(mu, dtype=float)
    sigma_arr = np.asarray(sigma, dtype=float)

    if mu_arr.ndim != 1:
        raise ValueError("mu must be a one-dimensional vector")
    if sigma_arr.ndim != 2 or sigma_arr.shape[0] != sigma_arr.shape[1]:
        raise ValueError("sigma must be a square matrix")
    if sigma_arr.shape[0] != mu_arr.shape[0]:
        raise ValueError("mu and sigma dimensions do not match")
    if not 0 <= cardinality <= mu_arr.shape[0]:
        raise ValueError("cardinality must be between 0 and len(mu)")
    if risk_aversion < 0:
        raise ValueError("risk_aversion must be non-negative")
    if penalty is not None and penalty < 0:
        raise ValueError("penalty must be non-negative")
    if not np.allclose(sigma_arr, sigma_arr.T):
        raise ValueError("sigma must be symmetric")

    return mu_arr, sigma_arr


def portfolio_objective(
    x: Iterable[int] | np.ndarray,
    mu: Iterable[float] | np.ndarray,
    sigma: Iterable[Iterable[float]] | np.ndarray,
    risk_aversion: float,
) -> float:
    """Return q x' Sigma x - mu' x for a binary selection vector."""

    mu_arr, sigma_arr = _validate_inputs(
        mu, sigma, cardinality=0, risk_aversion=risk_aversion
    )
    x_arr = _as_binary_vector(x, n=len(mu_arr)).astype(float)
    return float(risk_aversion * x_arr @ sigma_arr @ x_arr - mu_arr @ x_arr)


def penalized_objective(
    x: Iterable[int] | np.ndarray,
    mu: Iterable[float] | np.ndarray,
    sigma: Iterable[Iterable[float]] | np.ndarray,
    cardinality: int,
    risk_aversion: float,
    penalty: float,
) -> float:
    """Return q x' Sigma x - mu' x + A (1' x - K)^2."""

    base = portfolio_objective(x, mu, sigma, risk_aversion)
    x_arr = _as_binary_vector(x)
    return float(base + penalty * (x_arr.sum() - cardinality) ** 2)


def build_cardinality_qubo(
    mu: Iterable[float] | np.ndarray,
    sigma: Iterable[Iterable[float]] | np.ndarray,
    cardinality: int,
    risk_aversion: float,
    penalty: float,
) -> tuple[Qubo, float]:
    """Build the upper-triangular QUBO dictionary and constant offset.

    The returned dictionary follows the convention

        E(x) = sum_{i <= j} Q[i, j] x_i x_j + offset.

    For symmetric Sigma:

        Q[i, i] = q Sigma[i, i] - mu[i] + A (1 - 2K)
        Q[i, j] = 2q Sigma[i, j] + 2A, for i < j
        offset = A K^2
    """

    mu_arr, sigma_arr = _validate_inputs(
        mu, sigma, cardinality, risk_aversion, penalty
    )
    n = len(mu_arr)
    qubo: Qubo = {}

    for i in range(n):
        qubo[(i, i)] = float(
            risk_aversion * sigma_arr[i, i]
            - mu_arr[i]
            + penalty * (1 - 2 * cardinality)
        )
        for j in range(i + 1, n):
            qubo[(i, j)] = float(2 * risk_aversion * sigma_arr[i, j] + 2 * penalty)

    offset = float(penalty * cardinality**2)
    return qubo, offset


def qubo_energy(
    x: Iterable[int] | np.ndarray,
    qubo: Qubo,
    offset: float = 0.0,
) -> float:
    """Evaluate an upper-triangular QUBO dictionary."""

    if not qubo:
        x_arr = _as_binary_vector(x)
        return float(offset + 0 * x_arr.sum())

    max_index = max(max(i, j) for i, j in qubo)
    x_arr = _as_binary_vector(x, n=max_index + 1)
    energy = float(offset)
    for (i, j), coefficient in qubo.items():
        if i > j:
            raise ValueError("qubo must use upper-triangular keys with i <= j")
        energy += float(coefficient) * int(x_arr[i]) * int(x_arr[j])
    return float(energy)


def exact_cardinality_optimum(
    mu: Iterable[float] | np.ndarray,
    sigma: Iterable[Iterable[float]] | np.ndarray,
    cardinality: int,
    risk_aversion: float,
) -> BinarySolution:
    """Find the exact constrained optimum by enumerating K-combinations."""

    mu_arr, sigma_arr = _validate_inputs(
        mu, sigma, cardinality, risk_aversion
    )
    n = len(mu_arr)
    best: BinarySolution | None = None

    for selected in combinations(range(n), cardinality):
        x = np.zeros(n, dtype=int)
        x[list(selected)] = 1
        value = portfolio_objective(x, mu_arr, sigma_arr, risk_aversion)
        if best is None or value < best.value:
            best = BinarySolution(x=x, value=value)

    if best is None:
        raise RuntimeError("no feasible solution found")
    return best


def exact_qubo_optimum(
    qubo: Qubo,
    offset: float = 0.0,
) -> BinarySolution:
    """Find the exact QUBO optimum by enumerating all binary vectors."""

    if not qubo:
        return BinarySolution(x=np.zeros(0, dtype=int), value=float(offset))

    n = max(max(i, j) for i, j in qubo) + 1
    best: BinarySolution | None = None

    for bits in product((0, 1), repeat=n):
        x = np.asarray(bits, dtype=int)
        value = qubo_energy(x, qubo, offset)
        if best is None or value < best.value:
            best = BinarySolution(x=x, value=value)

    if best is None:
        raise RuntimeError("no QUBO solution found")
    return best
