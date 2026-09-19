from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .generative_data import atomic_save_numpy


def teacher_transition(vocabulary_size: int = 12, seed: int = 17) -> np.ndarray:
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(vocabulary_size, vocabulary_size))
    logits += np.eye(vocabulary_size) * 0.8
    values = np.exp(logits - logits.max(axis=1, keepdims=True))
    return values / values.sum(axis=1, keepdims=True)


def initial_transition(teacher: np.ndarray, severity: float = 0.65) -> np.ndarray:
    shifted = np.roll(teacher, 1, axis=1)
    result = (1 - severity) * teacher + severity * shifted
    return result / result.sum(axis=1, keepdims=True)


def sample_markov(
    transition: np.ndarray,
    count: int,
    length: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vocabulary = transition.shape[0]
    rows = np.empty((count, length), dtype=np.int32)
    rows[:, 0] = rng.integers(0, vocabulary, size=count)
    for position in range(1, length):
        uniforms = rng.random(count)
        cumulative = np.cumsum(transition[rows[:, position - 1]], axis=1)
        rows[:, position] = (uniforms[:, None] > cumulative).sum(axis=1)
    return rows


def fit_markov(sequences: np.ndarray, smoothing: float = 0.25) -> np.ndarray:
    vocabulary = int(sequences.max()) + 1
    counts = np.full((vocabulary, vocabulary), smoothing, dtype=np.float64)
    for row in sequences:
        np.add.at(counts, (row[:-1], row[1:]), 1)
    return counts / counts.sum(axis=1, keepdims=True)


def save_transition(path: str | Path, transition: np.ndarray) -> None:
    atomic_save_numpy(path, transition.astype(np.float64))


def load_transition(path: str | Path) -> np.ndarray:
    return np.load(path, allow_pickle=False)


def markov_metrics(teacher: np.ndarray, student: np.ndarray) -> dict[str, Any]:
    vocabulary = teacher.shape[0]
    stationary = np.full(vocabulary, 1 / vocabulary)
    for _ in range(10_000):
        updated = stationary @ teacher
        if np.max(np.abs(updated - stationary)) < 1e-14:
            break
        stationary = updated
    conditional_kl = np.sum(
        teacher * (np.log(teacher + 1e-300) - np.log(student + 1e-300)), axis=1
    )
    kl_rate = float(stationary @ conditional_kl)
    cross_entropy = float(-np.sum(stationary[:, None] * teacher * np.log(student + 1e-300)))
    teacher_entropy = float(-np.sum(stationary[:, None] * teacher * np.log(teacher + 1e-300)))
    return {
        "teacher_student_kl_per_token": kl_rate,
        "student_cross_entropy": cross_entropy,
        "teacher_entropy": teacher_entropy,
        "perplexity": float(np.exp(cross_entropy)),
    }

