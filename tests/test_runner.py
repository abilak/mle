from dataclasses import replace
from pathlib import Path

import pytest

from grounding_mle.config import load_config
from grounding_mle.io import append_jsonl, atomic_write_json, read_jsonl
from grounding_mle.planning import plan_config
from grounding_mle.runner import run_planned


ROOT = Path(__file__).resolve().parents[1]


def test_analytic_end_to_end_run_is_resumable(tmp_path):
    config = load_config(ROOT / "configs" / "smoke.yaml")
    planned = plan_config(config)[0]
    state = run_planned(planned, tmp_path)
    assert state["status"] == "complete"
    assert len(state["actual_real"]) == 3
    assert state["corpus_examples"] == 12
    repeated = run_planned(planned, tmp_path)
    assert repeated["corpus_hash"] == state["corpus_hash"]


def test_resume_discards_an_uncommitted_corpus_tail(tmp_path):
    config = load_config(ROOT / "configs" / "smoke.yaml")
    planned = plan_config(config)[0]
    state = run_planned(planned, tmp_path)
    run_dir = tmp_path / planned.run_id
    corpus_path = run_dir / "corpus.jsonl"
    committed = list(read_jsonl(corpus_path))
    append_jsonl(corpus_path, [committed[-1]])
    state["status"] = "running"
    atomic_write_json(run_dir / "state.json", state)

    resumed = run_planned(planned, tmp_path)

    assert resumed["status"] == "complete"
    assert len(list(read_jsonl(corpus_path))) == len(committed)


def test_resume_discards_artifacts_from_an_uncommitted_round(tmp_path):
    config = load_config(ROOT / "configs" / "smoke.yaml")
    planned = plan_config(config)[0]
    state = run_planned(planned, tmp_path)
    run_dir = tmp_path / planned.run_id
    corpus_path = run_dir / "corpus.jsonl"

    # Roll the atomic state back to round two while retaining the corpus tail and
    # round-three outputs, reproducing a crash between training and state commit.
    state["status"] = "running"
    state["next_round"] = 2
    state["current_model"] = state["trajectory"][2]["checkpoint"]
    state["actual_real"] = state["actual_real"][:2]
    state["actual_synthetic"] = state["actual_synthetic"][:2]
    state["real_cursor"] = sum(state["actual_real"])
    state["synthetic_cursor"] = sum(state["actual_synthetic"])
    state["trajectory"] = state["trajectory"][:3]
    atomic_write_json(run_dir / "state.json", state)

    checkpoint_marker = run_dir / "checkpoints" / "round_03" / "orphan.txt"
    evaluation_marker = run_dir / "evaluation" / "round_03" / "orphan.txt"
    checkpoint_marker.write_text("stale", encoding="utf-8")
    evaluation_marker.write_text("stale", encoding="utf-8")

    resumed = run_planned(planned, tmp_path)

    assert resumed["status"] == "complete"
    assert not checkpoint_marker.exists()
    assert not evaluation_marker.exists()
    assert len(list(read_jsonl(corpus_path))) == resumed["corpus_examples"]


def test_resume_discards_first_round_tail_when_state_is_missing(tmp_path):
    config = load_config(ROOT / "configs" / "smoke.yaml")
    planned = plan_config(config)[0]
    state = run_planned(planned, tmp_path)
    run_dir = tmp_path / planned.run_id
    corpus_path = run_dir / "corpus.jsonl"
    (run_dir / "state.json").unlink()

    resumed = run_planned(planned, tmp_path)

    assert resumed["status"] == "complete"
    assert len(list(read_jsonl(corpus_path))) == state["corpus_examples"]


def test_resume_rejects_plan_mismatch_for_existing_run_id(tmp_path):
    config = load_config(ROOT / "configs" / "smoke.yaml")
    planned = plan_config(config)[0]
    run_planned(planned, tmp_path)

    with pytest.raises(RuntimeError, match="does not match plan entry"):
        run_planned(replace(planned, condition="changed"), tmp_path)
