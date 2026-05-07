"""Small classical baselines for sparse portfolio QUBO experiments."""

from __future__ import annotations

import math
from typing import Iterable

import dimod
import neal
import numpy as np

from sparse_portfolio.qubo import BinarySolution, Qubo, portfolio_objective, qubo_energy


def top_k_by_mu(
    mu: Iterable[float] | np.ndarray,
    sigma: Iterable[Iterable[float]] | np.ndarray,
    cardinality: int,
    risk_aversion: float,
) -> BinarySolution:
    """Select the K largest expected-return assets."""

    mu_arr = np.asarray(mu, dtype=float)
    if not 0 <= cardinality <= len(mu_arr):
        raise ValueError("cardinality must be between 0 and len(mu)")
    selected = np.argsort(mu_arr)[-cardinality:] if cardinality else []
    x = np.zeros(len(mu_arr), dtype=int)
    x[selected] = 1
    value = portfolio_objective(x, mu, sigma, risk_aversion)
    return BinarySolution(x=x, value=value)


def greedy_forward_selection(
    mu: Iterable[float] | np.ndarray,
    sigma: Iterable[Iterable[float]] | np.ndarray,
    cardinality: int,
    risk_aversion: float,
) -> BinarySolution:
    """Build a K-asset portfolio by greedy one-step objective improvement."""

    mu_arr = np.asarray(mu, dtype=float)
    if not 0 <= cardinality <= len(mu_arr):
        raise ValueError("cardinality must be between 0 and len(mu)")

    x = np.zeros(len(mu_arr), dtype=int)
    remaining = set(range(len(mu_arr)))

    for _ in range(cardinality):
        best_candidate: tuple[float, int] | None = None
        for asset in remaining:
            candidate = x.copy()
            candidate[asset] = 1
            value = portfolio_objective(candidate, mu, sigma, risk_aversion)
            if best_candidate is None or value < best_candidate[0]:
                best_candidate = (value, asset)
        if best_candidate is None:
            break
        _, asset = best_candidate
        x[asset] = 1
        remaining.remove(asset)

    return BinarySolution(x=x, value=portfolio_objective(x, mu, sigma, risk_aversion))


def local_search_swaps(
    mu: Iterable[float] | np.ndarray,
    sigma: Iterable[Iterable[float]] | np.ndarray,
    cardinality: int,
    risk_aversion: float,
    initial_x: Iterable[int] | np.ndarray | None = None,
    max_passes: int = 100,
    tolerance: float = 1e-12,
) -> BinarySolution:
    """Improve a K-asset portfolio with cardinality-preserving swaps."""

    mu_arr = np.asarray(mu, dtype=float)
    n = len(mu_arr)
    if not 0 <= cardinality <= n:
        raise ValueError("cardinality must be between 0 and len(mu)")

    if initial_x is None:
        x = greedy_forward_selection(mu, sigma, cardinality, risk_aversion).x.copy()
    else:
        x = np.asarray(initial_x, dtype=int).copy()
        if x.shape != (n,):
            raise ValueError("initial_x has the wrong length")
        if not np.all((x == 0) | (x == 1)):
            raise ValueError("initial_x must be binary")
        if int(x.sum()) != cardinality:
            raise ValueError("initial_x must have the requested cardinality")

    value = portfolio_objective(x, mu, sigma, risk_aversion)

    for _ in range(max_passes):
        best_x = x
        best_value = value
        selected = np.flatnonzero(x == 1)
        unselected = np.flatnonzero(x == 0)

        for out_asset in selected:
            for in_asset in unselected:
                candidate = x.copy()
                candidate[out_asset] = 0
                candidate[in_asset] = 1
                candidate_value = portfolio_objective(
                    candidate, mu, sigma, risk_aversion
                )
                if candidate_value < best_value - tolerance:
                    best_x = candidate
                    best_value = candidate_value

        if best_value >= value - tolerance:
            break
        x = best_x
        value = best_value

    return BinarySolution(x=x, value=value)


def simulated_annealing_qubo(
    qubo: Qubo,
    offset: float = 0.0,
    *,
    num_reads: int = 32,
    num_steps: int = 1_000,
    seed: int | None = 0,
    initial_temperature: float = 1.0,
    final_temperature: float = 1e-3,
) -> BinarySolution:
    """A small local simulated annealer for QUBO dictionaries."""

    if not qubo:
        return BinarySolution(x=np.zeros(0, dtype=int), value=float(offset))
    if num_reads <= 0:
        raise ValueError("num_reads must be positive")
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    if initial_temperature <= 0 or final_temperature <= 0:
        raise ValueError("temperatures must be positive")

    rng = np.random.default_rng(seed)
    n = max(max(i, j) for i, j in qubo) + 1
    best: BinarySolution | None = None

    for _ in range(num_reads):
        x = rng.integers(0, 2, size=n, dtype=int)
        value = qubo_energy(x, qubo, offset)

        for step in range(num_steps):
            if num_steps == 1:
                temperature = final_temperature
            else:
                progress = step / (num_steps - 1)
                temperature = initial_temperature * (
                    final_temperature / initial_temperature
                ) ** progress

            index = int(rng.integers(0, n))
            candidate = x.copy()
            candidate[index] = 1 - candidate[index]
            candidate_value = qubo_energy(candidate, qubo, offset)
            delta = candidate_value - value

            if delta <= 0 or rng.random() < math.exp(-delta / temperature):
                x = candidate
                value = candidate_value

        if best is None or value < best.value:
            best = BinarySolution(x=x.copy(), value=float(value))

    if best is None:
        raise RuntimeError("simulated annealing produced no solution")
    return best


def dimod_exact_qubo(
    qubo: Qubo,
    offset: float = 0.0,
) -> BinarySolution:
    """Solve a QUBO exactly using D-Wave Ocean's `dimod.ExactSolver`."""

    if not qubo:
        return BinarySolution(x=np.zeros(0, dtype=int), value=float(offset))

    bqm = dimod.BinaryQuadraticModel.from_qubo(qubo, offset=offset)
    sample_set = dimod.ExactSolver().sample(bqm)
    best = sample_set.first
    n = max(max(i, j) for i, j in qubo) + 1
    x = np.array([best.sample[i] for i in range(n)], dtype=int)
    return BinarySolution(x=x, value=float(best.energy))


def neal_simulated_annealing_qubo(
    qubo: Qubo,
    offset: float = 0.0,
    *,
    num_reads: int = 64,
    num_sweeps: int = 1_000,
    seed: int | None = 0,
    beta_range: tuple[float, float] | None = None,
) -> BinarySolution:
    """Solve a QUBO approximately using D-Wave `neal` simulated annealing."""

    if not qubo:
        return BinarySolution(x=np.zeros(0, dtype=int), value=float(offset))

    bqm = dimod.BinaryQuadraticModel.from_qubo(qubo, offset=offset)
    sample_set = neal.SimulatedAnnealingSampler().sample(
        bqm,
        num_reads=num_reads,
        num_sweeps=num_sweeps,
        seed=seed,
        beta_range=beta_range,
    )
    best = sample_set.first
    n = max(max(i, j) for i, j in qubo) + 1
    x = np.array([best.sample[i] for i in range(n)], dtype=int)
    return BinarySolution(x=x, value=float(best.energy))
