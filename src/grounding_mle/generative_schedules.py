from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .theory import schedule_statistics


@dataclass(frozen=True)
class GenerativeSchedule:
    """A fully materialized real/synthetic schedule for a neural experiment."""

    name: str
    real_counts: tuple[int, ...]
    synthetic_counts: tuple[int, ...]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "real_counts": list(self.real_counts),
            "synthetic_counts": list(self.synthetic_counts),
            "metadata": self.metadata,
            "theory": schedule_statistics(
                self.real_counts, self.synthetic_counts
            ).to_dict(),
        }


def _largest_remainder(total: int, weights: Sequence[float], caps: Sequence[int]) -> list[int]:
    if total < 0:
        raise ValueError("total must be non-negative")
    if len(weights) != len(caps) or not weights:
        raise ValueError("weights and caps must be non-empty and have equal length")
    if total > sum(caps):
        raise ValueError("total exceeds schedule capacity")
    positive = [max(0.0, float(value)) for value in weights]
    if sum(positive) == 0:
        positive = [1.0] * len(positive)
    scale = total / sum(positive)
    raw = [value * scale for value in positive]
    result = [min(int(math.floor(value)), cap) for value, cap in zip(raw, caps, strict=True)]
    remaining = total - sum(result)
    order = sorted(
        range(len(result)),
        key=lambda index: (raw[index] - math.floor(raw[index]), positive[index], -index),
        reverse=True,
    )
    while remaining:
        progressed = False
        for index in order:
            if result[index] < caps[index]:
                result[index] += 1
                remaining -= 1
                progressed = True
                if not remaining:
                    break
        if not progressed:
            raise RuntimeError("unable to allocate the requested schedule total")
    return result


def _fraction_schedule(
    rounds: int,
    batch_size: int,
    kind: str,
    specification: Mapping[str, Any],
) -> list[int]:
    rounds_one_based = range(1, rounds + 1)
    if kind == "constant_fraction":
        fractions = [float(specification["fraction"])] * rounds
    elif kind == "log_fraction":
        power = float(specification.get("power", 1.0))
        anchor = float(specification.get("initial_fraction", 0.2))
        offset = float(specification.get("offset", 2.0))
        normalizer = math.log(1 + offset) ** power
        fractions = [
            anchor * normalizer / (math.log(t + offset) ** power)
            for t in rounds_one_based
        ]
    elif kind == "power_fraction":
        power = float(specification.get("power", 0.5))
        anchor = float(specification.get("initial_fraction", 0.2))
        fractions = [anchor / (t**power) for t in rounds_one_based]
    elif kind == "exponential_fraction":
        anchor = float(specification.get("initial_fraction", 0.2))
        decay = float(specification.get("decay", 0.85))
        fractions = [anchor * decay ** (t - 1) for t in rounds_one_based]
    elif kind == "no_real":
        fractions = [0.0] * rounds
    elif kind == "all_real":
        fractions = [1.0] * rounds
    else:
        raise ValueError(f"Unknown fraction schedule kind: {kind}")
    if any(not 0 <= value <= 1 for value in fractions):
        raise ValueError(f"Schedule {kind} produced a fraction outside [0, 1]")
    return [min(batch_size, max(0, int(round(batch_size * value)))) for value in fractions]


def _matched_budget_schedule(
    rounds: int,
    batch_size: int,
    kind: str,
    specification: Mapping[str, Any],
) -> list[int]:
    total_real = int(specification["total_real"])
    caps = [batch_size] * rounds
    if kind == "uniform_budget":
        weights = [1.0] * rounds
    elif kind == "front_loaded":
        weights = [float(rounds - index) for index in range(rounds)]
        if specification.get("block", False):
            full, remainder = divmod(total_real, batch_size)
            result = [batch_size] * full
            if len(result) < rounds:
                result.append(remainder)
            return (result + [0] * rounds)[:rounds]
    elif kind == "back_loaded":
        weights = [float(index + 1) for index in range(rounds)]
        if specification.get("block", False):
            full, remainder = divmod(total_real, batch_size)
            suffix = ([remainder] if remainder else []) + [batch_size] * full
            return ([0] * rounds + suffix)[-rounds:]
    elif kind == "middle_loaded":
        midpoint = (rounds - 1) / 2
        weights = [1 / (1 + abs(index - midpoint)) for index in range(rounds)]
    elif kind == "alternating":
        weights = [1.0 if index % 2 == 0 else 0.05 for index in range(rounds)]
    elif kind == "bursty":
        period = max(2, int(specification.get("period", 5)))
        weights = [1.0 if index % period == 0 else 0.0 for index in range(rounds)]
    else:
        raise ValueError(f"Unknown matched-budget schedule kind: {kind}")
    return _largest_remainder(total_real, weights, caps)


def materialize_generative_schedule(
    name: str,
    specification: Mapping[str, Any],
    rounds: int,
    default_batch_size: int,
) -> GenerativeSchedule:
    """Resolve a declarative schedule while preserving exact matched budgets."""

    if rounds <= 0:
        raise ValueError("rounds must be positive")
    batch_size = int(specification.get("batch_size", default_batch_size))
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    kind = str(specification.get("kind", "constant_fraction"))
    if kind == "explicit":
        real = [int(value) for value in specification["real"]]
        synthetic = [int(value) for value in specification["synthetic"]]
        if len(real) != rounds or len(synthetic) != rounds:
            raise ValueError(f"Explicit schedule {name} must contain {rounds} rounds")
    elif kind in {
        "uniform_budget",
        "front_loaded",
        "back_loaded",
        "middle_loaded",
        "alternating",
        "bursty",
    }:
        real = _matched_budget_schedule(rounds, batch_size, kind, specification)
        synthetic = [batch_size - value for value in real]
    else:
        real = _fraction_schedule(rounds, batch_size, kind, specification)
        synthetic = [batch_size - value for value in real]
    if any(value < 0 for value in real + synthetic):
        raise ValueError(f"Schedule {name} contains a negative count")
    return GenerativeSchedule(
        name=name,
        real_counts=tuple(real),
        synthetic_counts=tuple(synthetic),
        metadata={
            "kind": kind,
            "batch_size": batch_size,
            "requested": dict(specification),
        },
    )

