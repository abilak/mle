#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize completion of a grounding-mle generative plan."
    )
    parser.add_argument(
        "--plan", default="runs/generative_confirmatory_plan.json"
    )
    parser.add_argument("--runs", default="runs/generative")
    args = parser.parse_args()

    plan_path = Path(args.plan)
    runs_root = Path(args.runs)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("kind") != "grounding-mle-generative-plan-v1":
        raise SystemExit(f"Not a generative plan: {plan_path}")

    counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"planned": 0, "complete": 0, "partial": 0, "missing": 0}
    )
    pending_indices: list[int] = []
    for index, run in enumerate(plan["runs"]):
        experiment = str(run["experiment"])
        counts[experiment]["planned"] += 1
        state_path = runs_root / str(run["run_id"]) / "state.json"
        if not state_path.exists():
            counts[experiment]["missing"] += 1
            pending_indices.append(index)
            continue
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("status") == "complete":
            counts[experiment]["complete"] += 1
        else:
            counts[experiment]["partial"] += 1
            pending_indices.append(index)

    header = f"{'experiment':34s} {'planned':>7s} {'complete':>8s} {'partial':>7s} {'missing':>7s}"
    print(header)
    print("-" * len(header))
    for experiment in sorted(counts):
        row = counts[experiment]
        print(
            f"{experiment:34s} {row['planned']:7d} {row['complete']:8d} "
            f"{row['partial']:7d} {row['missing']:7d}"
        )
    print("-" * len(header))
    complete = sum(row["complete"] for row in counts.values())
    planned = sum(row["planned"] for row in counts.values())
    print(f"complete: {complete}/{planned}")
    if pending_indices:
        preview = ", ".join(str(value) for value in pending_indices[:30])
        suffix = " ..." if len(pending_indices) > 30 else ""
        print(f"pending plan indices ({len(pending_indices)}): {preview}{suffix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
