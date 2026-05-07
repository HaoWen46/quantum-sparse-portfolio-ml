import numpy as np

from sparse_portfolio import (
    benchmark_solvers,
    generate_synthetic_instance,
    penalty_scale,
    run_penalty_sweep,
    summarize_penalty_sweep,
)


def test_generate_synthetic_instance_has_psd_covariance():
    instance = generate_synthetic_instance(n_assets=8, cardinality=3, seed=42)

    assert instance.mu.shape == (8,)
    assert instance.sigma.shape == (8, 8)
    assert np.allclose(instance.sigma, instance.sigma.T)
    assert np.linalg.eigvalsh(instance.sigma).min() >= -1e-10


def test_penalty_scale_is_positive():
    instance = generate_synthetic_instance(n_assets=5, cardinality=2, seed=1)

    assert penalty_scale(instance.mu, instance.sigma, instance.risk_aversion) > 0


def test_run_penalty_sweep_has_expected_rows_and_summary():
    sweep = run_penalty_sweep(
        n_instances=3,
        n_assets=6,
        cardinality=2,
        penalty_multipliers=(0.0, 0.1, 1.0),
        seed=10,
    )
    summary = summarize_penalty_sweep(sweep)

    assert len(sweep) == 9
    assert set(summary["penalty_multiplier"]) == {0.0, 0.1, 1.0}
    assert summary["feasibility_rate"].between(0.0, 1.0).all()


def test_benchmark_solvers_contains_core_solvers():
    instance = generate_synthetic_instance(n_assets=6, cardinality=2, seed=5)

    result = benchmark_solvers(
        instance,
        penalty_multiplier=1.0,
        annealing_reads=16,
        annealing_steps=200,
        seed=2,
    )

    assert set(result["solver"]) == {
        "exact_constrained",
        "exact_qubo",
        "dimod_exact_qubo",
        "top_k_mu",
        "greedy",
        "local_search",
        "annealing_qubo",
        "neal_qubo",
    }
    assert result.loc[result["solver"] == "exact_constrained", "gap_vs_exact"].iloc[0] == 0
