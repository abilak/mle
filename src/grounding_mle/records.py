from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class TrainingExample:
    example_id: str
    prompt: str
    response: str
    source: str
    round_index: int
    task_id: str
    tests: list[str] = field(default_factory=list)
    skill: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PromptRecord:
    task_id: str
    prompt: str
    human_responses: list[str]
    tests: list[str] = field(default_factory=list)
    skill: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "PromptRecord":
        return cls(
            task_id=str(row["task_id"]),
            prompt=str(row["prompt"]),
            human_responses=[str(x) for x in row.get("human_responses", [])],
            tests=[str(x) for x in row.get("tests", [])],
            skill=str(row.get("skill", "unknown")),
            metadata=dict(row.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

