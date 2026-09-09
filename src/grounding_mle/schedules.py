from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from scipy.optimize import minimize

from .theory import risk_proxy, schedule_statistics


def _apportion(weights: Iterable[float], total: int) -> list[int]:
    weights_array = np.asarray(tuple(weights), dtype=float)
    if total < 0 or len(weights_array) == 0 or np.any(weights_array < 0):
        raise ValueError("Invalid apportionment inputs")
    if total == 0:
        return [0] * len(weights_array)
    if weights_array.sum() <= 0:
        raise ValueError("At least one schedule weight must be positive")
    quota = total * weights_array / weights_array.sum()
    allocation = np.floor(quota).astype(int)
    remainder = total - int(allocation.sum())
    order = np.argsort(-(quota - allocation), kind="stable")
    allocation[order[:remainder]] += 1
    return [int(x) for x in allocation]


def fixed_schedule(kind: str, rounds: int, total_real: int, **kwargs: Any) -> list[int]:
    if rounds <= 0:
        raise ValueError("rounds must be positive")
    index = np.arange(rounds, dtype=float)
    if kind == "uniform":
        weights = np.ones(rounds)
    elif kind == "front_loaded":
        weights = np.exp(-float(kwargs.get("decay", 0.55)) * index)
    elif kind == "back_loaded":
        weights = np.exp(float(kwargs.get("growth", 0.55)) * index)
    elif kind == "decaying":
        weights = np.power(index + 1.0, -float(kwargs.get("power", 1.0)))
    elif kind == "growing":
        weights = np.power(index + 1.0, float(kwargs.get("power", 1.0)))
    elif kind == "periodic":
        every = int(kwargs.get("every", 3))
        offset = int(kwargs.get("offset", every - 1))
        weights = ((index.astype(int) - offset) % every == 0).astype(float)
        if not weights.any():
            weights[-1] = 1.0
    elif kind == "bursty":
        seed = int(kwargs.get("seed", 0))
        concentration = float(kwargs.get("concentration", 0.25))
        weights = np.random.default_rng(seed).dirichlet(np.full(rounds, concentration))
    elif kind == "initial_only":
        weights = np.zeros(rounds)
        weights[0] = 1.0
    else:
        raise ValueError(f"Unknown fixed schedule kind: {kind}")
    return _apportion(weights, total_real)


def synthetic_schedule(rounds: int, per_round: int | list[int]) -> list[int]:
    if isinstance(per_round, int):
        if per_round < 0:
            raise ValueError("synthetic_per_round must be non-negative")
        return [per_round] * rounds
    values = [int(x) for x in per_round]
    if len(values) != rounds or any(x < 0 for x in values):
        raise ValueError("Synthetic schedule has invalid length or negative counts")
    return values


def vanishing_external_schedule(
    rounds: int,
    synthetic_scale: float,
    log_power: float,
    real_scale: float = 1.0,
) -> tuple[list[int], list[int]]:
    """Paper-aligned power-log schedule with n_t ~ t and m_t ~ t/log(t)^r.

    Both batch real fractions vanish. The restoring mass diverges at ``r <= 1``
    and converges at ``r > 1`` under the asymptotic continuous schedule.
    """

    t = np.arange(2, rounds + 2, dtype=float)
    synthetic = np.maximum(1, np.rint(synthetic_scale * t)).astype(int)
    real = np.maximum(1, np.rint(real_scale * t / np.log(t) ** log_power)).astype(int)
    return real.tolist(), synthetic.tolist()


def random_schedule_bank(
    rounds: int,
    total_real: int,
    synthetic_counts: list[int],
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Create a diverse, deterministic, matched-budget schedule bank."""

    rng = np.random.default_rng(seed)
    schedules: list[dict[str, Any]] = []
    seen: set[tuple[int, ...]] = set()
    templates: list[tuple[str, dict[str, Any]]] = [
        ("uniform", {}),
        ("front_loaded", {"decay": 0.8}),
        ("back_loaded", {"growth": 0.8}),
        ("periodic", {"every": 2}),
        ("periodic", {"every": 3}),
        ("decaying", {"power": 1.0}),
        ("growing", {"power": 1.0}),
    ]
    for name, options in templates:
        values = fixed_schedule(name, rounds, total_real, **options)
        key = tuple(values)
        if key not in seen:
            seen.add(key)
            schedules.append({"name": f"{name}_{len(schedules):02d}", "real": values})
    concentrations = np.geomspace(0.08, 4.0, max(count * 2, 20))
    attempt = 0
    while len(schedules) < count and attempt < count * 100:
        alpha = float(concentrations[attempt % len(concentrations)])
        weights = rng.dirichlet(np.full(rounds, alpha))
        values = _apportion(weights, total_real)
        key = tuple(values)
        if key not in seen:
            seen.add(key)
            schedules.append({"name": f"random_{len(schedules):02d}", "real": values})
        attempt += 1
    if len(schedules) != count:
        raise RuntimeError("Could not create the requested number of unique schedules")
    order = rng.permutation(count)
    train_count = count // 2
    train_indices = set(int(x) for x in order[:train_count])
    for index, schedule in enumerate(schedules):
        schedule["synthetic"] = list(synthetic_counts)
        schedule["law_split"] = "train" if index in train_indices else "test"
        schedule["theory"] = schedule_statistics(
            schedule["real"], schedule["synthetic"]
        ).final
    return schedules


@dataclass(frozen=True)
class ControllerResult:
    real_counts: tuple[int, ...]
    risk: float
    success: bool
    message: str


def optimize_fixed_budget(
    synthetic_counts: Iterable[int],
    budget: int,
    bias_scale: float = 1.0,
    noise_scale: float = 1.0,
    restarts: int = 12,
    seed: int = 0,
) -> ControllerResult:
    """Minimize the finite-horizon MLE risk proxy, then integerize exactly.

    Continuous SLSQP is used only to propose schedules. A deterministic local
    exchange search over integer schedules produces the returned allocation.
    """

    synthetic = np.asarray(tuple(synthetic_counts), dtype=int)
    rounds = len(synthetic)
    if rounds == 0 or budget < 0 or np.any(synthetic < 0):
        raise ValueError("Invalid controller inputs")
    if budget == 0:
        real = np.zeros(rounds, dtype=int)
        return ControllerResult(
            tuple(int(x) for x in real),
            risk_proxy(real, synthetic, bias_scale, noise_scale),
            True,
            "zero budget",
        )

    rng = np.random.default_rng(seed)

    def objective(x: np.ndarray) -> float:
        real = np.asarray(x, dtype=float)
        batch = real + synthetic
        cumulative = np.cumsum(batch)
        if np.any(batch <= 0) or np.any(cumulative <= 0):
            return float("inf")
        gamma = real / cumulative
        survival = float(np.prod(1.0 - gamma))
        propagated = 0.0
        for g, size, total in zip(gamma, batch, cumulative, strict=True):
            innovation = size / (total * total)
            propagated = (1.0 - g) ** 2 * propagated + innovation
        return float(bias_scale * survival**2 + noise_scale * propagated)

    bounds = [(0.0, float(budget))] * rounds
    constraint = {"type": "eq", "fun": lambda x: float(x.sum() - budget)}
    candidates: list[np.ndarray] = [np.full(rounds, budget / rounds)]
    candidates.extend(rng.dirichlet(np.ones(rounds), size=max(0, restarts - 1)) * budget)
    best_x = candidates[0]
    best_value = float("inf")
    best_message = ""
    success = False
    for x0 in candidates:
        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=[constraint],
            options={"maxiter": 800, "ftol": 1e-12},
        )
        if float(result.fun) < best_value:
            best_value = float(result.fun)
            best_x = np.asarray(result.x)
            best_message = str(result.message)
            success = bool(result.success)

    integer = np.asarray(_apportion(np.maximum(best_x, 0.0), budget), dtype=int)
    current = objective(integer)
    improved = True
    while improved:
        improved = False
        for source in range(rounds):
            if integer[source] == 0:
                continue
            for target in range(rounds):
                if source == target:
                    continue
                proposal = integer.copy()
                proposal[source] -= 1
                proposal[target] += 1
                value = objective(proposal)
                if value + 1e-15 < current:
                    integer, current, improved = proposal, value, True
                    break
            if improved:
                break
    assert int(integer.sum()) == budget and np.all(integer >= 0)
    return ControllerResult(
        tuple(int(x) for x in integer), float(current), success, best_message
    )


def minimum_budget_for_target(
    synthetic_counts: Iterable[int],
    target_risk: float,
    max_budget: int,
    **kwargs: Any,
) -> ControllerResult:
    if target_risk <= 0 or max_budget < 0:
        raise ValueError("target_risk and max_budget must be positive")
    synthetic = tuple(int(x) for x in synthetic_counts)
    low, high = 0, max_budget
    best: ControllerResult | None = None
    while low <= high:
        middle = (low + high) // 2
        result = optimize_fixed_budget(synthetic, middle, **kwargs)
        if result.risk <= target_risk:
            best = result
            high = middle - 1
        else:
            low = middle + 1
    if best is None:
        result = optimize_fixed_budget(synthetic, max_budget, **kwargs)
        return ControllerResult(result.real_counts, result.risk, False, "target not reached")
    return best
