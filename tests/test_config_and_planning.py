from pathlib import Path

from grounding_mle.config import load_config
from grounding_mle.planning import plan_config


ROOT = Path(__file__).resolve().parents[1]


def test_every_experiment_config_loads_and_plans():
    paths = sorted((ROOT / "configs" / "experiments").glob("*.yaml"))
    assert len(paths) == 10
    for path in paths:
        runs = plan_config(load_config(path))
        assert runs, path
        assert all(len(run.real_counts) == len(run.synthetic_counts) for run in runs)
        assert len({run.run_id for run in runs}) == len(runs)


def test_schedule_law_has_40_schedules_times_three_seeds():
    config = load_config(ROOT / "configs" / "experiments" / "02_schedule_law.yaml")
    runs = plan_config(config)
    assert len(runs) == 120
    assert {run.law_split for run in runs} == {"train", "test"}


def test_controller_has_every_preregistered_comparator_at_each_budget():
    config = load_config(ROOT / "configs" / "experiments" / "03_grounding_controller.yaml")
    runs = plan_config(config)
    conditions = {run.condition for run in runs}
    for budget in (256, 512, 768, 1024):
        assert {
            f"uniform_b{budget}",
            f"periodic_b{budget}",
            f"front_b{budget}",
            f"back_b{budget}",
            f"random_b{budget}",
            f"theory_b{budget}",
        }.issubset(conditions)
    assert "reactive_cap1024" in conditions


def test_scale_replication_includes_frozen_good_and_bad_schedules():
    config = load_config(
        ROOT / "configs" / "experiments" / "08_model_scale_and_architecture.yaml"
    )
    conditions = {run.condition for run in plan_config(config)}
    for prefix in ("qwen_05b", "qwen_15b", "deepseek_13b"):
        assert f"{prefix}_selected_bad" in conditions
        assert f"{prefix}_selected_good" in conditions
