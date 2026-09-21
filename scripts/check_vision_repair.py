#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from grounding_mle.generative_vision_repair import (
    evaluate_repair,
    load_repair_specification,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply strict absolute and relative gates to the vision repair."
    )
    parser.add_argument("--results", default="results/generative_vision_repair")
    parser.add_argument("--config", default="configs/generative/vision_repair.yaml")
    parser.add_argument("--output")
    args = parser.parse_args()

    results = Path(args.results)
    report = evaluate_repair(results, load_repair_specification(args.config))
    output = Path(args.output) if args.output else results / "quality_gate.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["quantitative_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
