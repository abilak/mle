from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class ScheduleStatistics:
    real_counts: tuple[int, ...]
    synthetic_counts: tuple[int, ...]
    batch_sizes: tuple[int, ...]
    cumulative_sizes: tuple[int, ...]
    batch_real_fractions: tuple[float, ...]
    gamma: tuple[float, ...]
    restoring_mass: tuple[float, ...]
    survival_product: tuple[float, ...]
    innovation_variance: tuple[float, ...]
    propagated_noise: tuple[float, ...]
    transformed_noise: tuple[float, ...]

    @property
    def final(self) -> dict[str, float]:
        return {
            "total_real": float(sum(self.real_counts)),
            "total_synthetic": float(sum(self.synthetic_counts)),
            "mean_batch_real_fraction": float(np.mean(self.batch_real_fractions)),
            "final_batch_real_fraction": float(self.batch_real_fractions[-1]),
            "cumulative_real_fraction": float(
                sum(self.real_counts) / sum(self.batch_sizes)
            ),
            "G_T": self.restoring_mass[-1],
            "Q_T": self.survival_product[-1],
            "Q_T_squared": self.survival_product[-1] ** 2,
            "Q_T_squared_A_T": self.propagated_noise[-1],
            "A_T": self.transformed_noise[-1],
        }

    def to_dict(self) -> dict[str, object]:
        return asdict(self) | {"final": self.final}


def _validate_counts(real_counts: Iterable[int], synthetic_counts: Iterable[int]) -> tuple[np.ndarray, np.ndarray]:
    real = np.asarray(tuple(real_counts), dtype=np.int64)
    synthetic = np.asarray(tuple(synthetic_counts), dtype=np.int64)
    if real.ndim != 1 or synthetic.ndim != 1 or len(real) == 0:
        raise ValueError("Schedules must be non-empty one-dimensional sequences")
    if len(real) != len(synthetic):
        raise ValueError("Real and synthetic schedules must have the same length")
    if np.any(real < 0) or np.any(synthetic < 0):
        raise ValueError("Sample counts must be non-negative")
    if np.any(real + synthetic <= 0):
        raise ValueError("Every round must add at least one observation")
    return real, synthetic


def schedule_statistics(
    real_counts: Iterable[int], synthetic_counts: Iterable[int]
) -> ScheduleStatistics:
    """Compute the finite-horizon schedule variables from the paper.

    The numerically stable ``propagated_noise`` recursion equals ``Q_t^2 A_t``
    whenever no gamma is one, and remains defined for all-real initial rounds.
    """

    real, synthetic = _validate_counts(real_counts, synthetic_counts)
    batch = real + synthetic
    cumulative = np.cumsum(batch)
    rho = real / batch
    gamma = real / cumulative
    restoring = np.cumsum(gamma)
    q = np.cumprod(1.0 - gamma)
    v = batch / np.square(cumulative.astype(float))

    propagated = np.empty_like(v, dtype=float)
    running = 0.0
    for t, (g, innovation) in enumerate(zip(gamma, v, strict=True)):
        running = (1.0 - g) ** 2 * running + innovation
        propagated[t] = running

    transformed = np.empty_like(v, dtype=float)
    for t, (qt, pt) in enumerate(zip(q, propagated, strict=True)):
        transformed[t] = pt / (qt * qt) if qt > 0 else float("inf")

    return ScheduleStatistics(
        real_counts=tuple(int(x) for x in real),
        synthetic_counts=tuple(int(x) for x in synthetic),
        batch_sizes=tuple(int(x) for x in batch),
        cumulative_sizes=tuple(int(x) for x in cumulative),
        batch_real_fractions=tuple(float(x) for x in rho),
        gamma=tuple(float(x) for x in gamma),
        restoring_mass=tuple(float(x) for x in restoring),
        survival_product=tuple(float(x) for x in q),
        innovation_variance=tuple(float(x) for x in v),
        propagated_noise=tuple(float(x) for x in propagated),
        transformed_noise=tuple(float(x) for x in transformed),
    )


def risk_proxy(
    real_counts: Iterable[int],
    synthetic_counts: Iterable[int],
    bias_scale: float = 1.0,
    noise_scale: float = 1.0,
) -> float:
    stats = schedule_statistics(real_counts, synthetic_counts)
    q2 = stats.survival_product[-1] ** 2
    return float(bias_scale * q2 + noise_scale * stats.propagated_noise[-1])


def gaussian_exact_mse(
    real_counts: Iterable[int],
    synthetic_counts: Iterable[int],
    initial_error: float = 1.0,
    variance: float = 1.0,
) -> np.ndarray:
    stats = schedule_statistics(real_counts, synthetic_counts)
    risk = initial_error**2
    path: list[float] = []
    for gamma, innovation in zip(
        stats.gamma, stats.innovation_variance, strict=True
    ):
        risk = (1.0 - gamma) ** 2 * risk + variance * innovation
        path.append(float(risk))
    return np.asarray(path)

