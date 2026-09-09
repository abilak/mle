import numpy as np

from grounding_mle.schedules import (
    fixed_schedule,
    optimize_fixed_budget,
    random_schedule_bank,
    vanishing_external_schedule,
)
from grounding_mle.theory import risk_proxy, schedule_statistics


def test_all_fixed_schedules_preserve_budget():
    for kind in [
        "uniform",
        "front_loaded",
        "back_loaded",
        "periodic",
        "decaying",
        "growing",
        "bursty",
        "initial_only",
    ]:
        schedule = fixed_schedule(kind, 10, 103, seed=7)
        assert len(schedule) == 10
        assert sum(schedule) == 103
        assert min(schedule) >= 0


def test_schedule_bank_is_unique_matched_and_blinded():
    bank = random_schedule_bank(10, 100, [50] * 10, count=40, seed=9)
    assert len({tuple(item["real"]) for item in bank}) == 40
    assert all(sum(item["real"]) == 100 for item in bank)
    assert {item["law_split"] for item in bank} == {"train", "test"}
    assert sum(item["law_split"] == "train" for item in bank) == 20


def test_controller_never_worse_than_uniform_after_integer_search():
    synthetic = [50] * 8
    result = optimize_fixed_budget(synthetic, budget=80, seed=3)
    uniform = fixed_schedule("uniform", 8, 80)
    assert sum(result.real_counts) == 80
    assert result.risk <= risk_proxy(uniform, synthetic) + 1e-10


def test_zero_budget_controller_returns_plain_integer_schedule():
    result = optimize_fixed_budget([50] * 4, budget=0, seed=3)
    assert result.real_counts == (0, 0, 0, 0)
    assert all(type(value) is int for value in result.real_counts)


def test_vanishing_schedules_have_different_restoring_growth():
    divergent_real, synthetic = vanishing_external_schedule(5000, 10, 1.0, 5)
    summable_real, _ = vanishing_external_schedule(5000, 10, 2.0, 5)
    divergent = schedule_statistics(divergent_real, synthetic)
    summable = schedule_statistics(summable_real, synthetic)
    assert divergent.batch_real_fractions[-1] < divergent.batch_real_fractions[50]
    assert summable.batch_real_fractions[-1] < summable.batch_real_fractions[50]
    assert divergent.restoring_mass[-1] > summable.restoring_mass[-1]
