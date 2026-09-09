import numpy as np
import pandas as pd
from pathlib import Path

from grounding_mle.analysis import (
    _capability_groups,
    analyze_runs,
    bootstrap_mean_ci,
    paired_effect,
)
from grounding_mle.config import load_config
from grounding_mle.planning import plan_config
from grounding_mle.runner import run_planned


ROOT = Path(__file__).resolve().parents[1]


def test_bootstrap_and_paired_effect_are_deterministic():
    first = bootstrap_mean_ci([0.5, 0.6, 0.7], seed=3)
    second = bootstrap_mean_ci([0.5, 0.6, 0.7], seed=3)
    assert first == second
    assert first[1] <= first[0] <= first[2]
    effect = paired_effect([0.8, 0.7, 0.9], [0.6, 0.5, 0.7])
    assert np.isclose(effect["paired_mean_difference"], 0.2)


def test_smoke_analysis_writes_paired_effects_and_figures(tmp_path):
    runs_dir = tmp_path / "runs"
    for planned in plan_config(load_config(ROOT / "configs" / "smoke.yaml")):
        run_planned(planned, runs_dir)
    result = analyze_runs(runs_dir, tmp_path / "analysis")
    assert result["completed_runs"] == 2
    assert result["paired_effect_rows"] == 1
    assert (tmp_path / "analysis" / "paired_effects.csv").exists()


def test_capability_groups_include_initial_tail_and_rarity():
    rows = []
    for task_id, skill, baseline_pass in [("weak", "rare_skill", False), ("strong", "common", True)]:
        for round_index, passed in [(0, baseline_pass), (1, baseline_pass)]:
            rows.append(
                {
                    "experiment": "exp05_capability_erosion",
                    "model": "model",
                    "run_id": "run",
                    "task_id": task_id,
                    "skill": skill,
                    "round": round_index,
                    "passed": passed,
                }
            )
    enriched = _capability_groups(pd.DataFrame(rows))
    weak = enriched[enriched["task_id"] == "weak"]
    assert set(weak["initial_tail"]) == {"bottom_decile"}
    assert "skill_rarity" in enriched
