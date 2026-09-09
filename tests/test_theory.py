import numpy as np

from grounding_mle.theory import gaussian_exact_mse, risk_proxy, schedule_statistics
from grounding_mle.theory_experiments import simulate_barycentric_mse


def test_schedule_statistics_match_hand_calculation():
    stats = schedule_statistics([1, 1], [1, 1])
    assert stats.cumulative_sizes == (2, 4)
    assert np.allclose(stats.gamma, [0.5, 0.25])
    assert np.allclose(stats.survival_product, [0.5, 0.375])
    assert np.allclose(stats.innovation_variance, [0.5, 0.125])
    assert np.isclose(stats.propagated_noise[-1], 0.40625)
    assert np.isclose(stats.final["Q_T_squared_A_T"], 0.40625)


def test_gaussian_exact_risk_recursion():
    exact = gaussian_exact_mse([10] * 30, [10] * 30)
    monte_carlo = simulate_barycentric_mse(
        "gaussian", [10] * 30, [10] * 30, paths=20_000, seed=4
    )
    assert np.all(exact > 0)
    assert np.isclose(exact[-1], monte_carlo[-1], rtol=0.08)


def test_real_data_reduces_bias_term():
    no_real = risk_proxy([0] * 5, [20] * 5, noise_scale=0)
    grounded = risk_proxy([5] * 5, [15] * 5, noise_scale=0)
    assert grounded < no_real


def test_all_real_round_is_numerically_defined():
    stats = schedule_statistics([20, 20], [0, 0])
    assert stats.survival_product[0] == 0
    assert np.isfinite(stats.propagated_noise[-1])
    assert np.isinf(stats.transformed_noise[-1])

