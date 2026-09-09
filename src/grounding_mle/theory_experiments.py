from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Callable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/grounding-mle-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from .io import atomic_write_json
from .theory import gaussian_exact_mse, schedule_statistics


PAPER_SEED = 20260821


def _draw_sufficient_sum(
    family: str, count: int, parameter: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    if count == 0:
        return np.zeros_like(parameter)
    if family == "gaussian":
        return rng.normal(count * parameter, math.sqrt(count))
    if family == "bernoulli":
        return rng.binomial(count, np.clip(parameter, 0.0, 1.0))
    if family == "poisson":
        return rng.poisson(count * np.maximum(parameter, 0.0))
    if family == "gamma":
        shape = 2.0
        return rng.gamma(count * shape, np.maximum(parameter, 1e-12) / shape)
    if family == "categorical":
        if parameter.ndim != 2 or parameter.shape[1] != 3:
            raise ValueError("categorical simulations currently expect three categories")
        result = np.zeros_like(parameter)
        result[:, 0] = rng.binomial(count, np.clip(parameter[:, 0], 0.0, 1.0))
        remaining = count - result[:, 0]
        denominator = np.maximum(1.0 - parameter[:, 0], 1e-12)
        conditional = np.clip(parameter[:, 1] / denominator, 0.0, 1.0)
        result[:, 1] = rng.binomial(remaining.astype(int), conditional)
        result[:, 2] = remaining - result[:, 1]
        return result
    raise ValueError(f"Unknown family: {family}")


def simulate_barycentric_mse(
    family: str,
    real_counts: list[int],
    synthetic_counts: list[int],
    paths: int,
    seed: int,
) -> np.ndarray:
    scalar_parameters = {
        "gaussian": (0.0, 1.0),
        # These are the paper's Figure 2 mean-coordinate normalizations:
        # initial squared error and the relevant variance scale are comparable.
        "bernoulli": (0.5, 1.0),
        "poisson": (0.0, 1.0),
        "gamma": (0.0, 1.0),
    }
    if family == "categorical":
        truth = np.broadcast_to(np.array([1 / 3, 1 / 3, 1 / 3]), (paths, 3)).copy()
        current = np.broadcast_to(np.array([1.0, 0.0, 0.0]), (paths, 3)).copy()
    else:
        truth_value, initial_value = scalar_parameters[family]
        truth = np.full(paths, truth_value, dtype=float)
        current = np.full(paths, initial_value, dtype=float)
    initial_error = np.mean(np.sum(np.square(current - truth), axis=-1)) if current.ndim == 2 else np.mean(np.square(current - truth))
    rng = np.random.default_rng(seed)
    cumulative_sum = np.zeros_like(current, dtype=float)
    cumulative_count = 0
    mse: list[float] = []
    for real_count, synthetic_count in zip(real_counts, synthetic_counts, strict=True):
        real_sum = _draw_sufficient_sum(family, real_count, truth, rng)
        synthetic_sum = _draw_sufficient_sum(family, synthetic_count, current, rng)
        cumulative_sum += real_sum + synthetic_sum
        cumulative_count += real_count + synthetic_count
        current = cumulative_sum / cumulative_count
        squared = np.sum(np.square(current - truth), axis=-1) if current.ndim == 2 else np.square(current - truth)
        mse.append(float(np.mean(squared) / initial_error))
    return np.asarray(mse)


def _save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def run_family_coverage(output: Path, profile: str) -> pd.DataFrame:
    rounds = 900 if profile == "full" else 180
    paths = 6000 if profile == "full" else 700
    families = ["bernoulli", "poisson", "gamma", "categorical"]
    schedules = {
        "divergent": ([25] * rounds, [25] * rounds),
        "summable": ([25] * rounds, [25 * (t + 1) ** 2 for t in range(rounds)]),
    }
    rows: list[dict[str, object]] = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for axis, (schedule_name, (real, synthetic)) in zip(axes, schedules.items(), strict=True):
        for family_index, family in enumerate(families):
            mse = simulate_barycentric_mse(
                family, real, synthetic, paths, PAPER_SEED + family_index
            )
            rounds_axis = np.arange(1, rounds + 1)
            axis.loglog(rounds_axis, mse, label=family.title())
            checkpoints = np.unique(np.geomspace(1, rounds, 30).astype(int))
            rows.extend(
                {
                    "experiment": "family_coverage",
                    "schedule": schedule_name,
                    "family": family,
                    "round": int(t),
                    "normalized_mse": float(mse[t - 1]),
                    "paths": paths,
                }
                for t in checkpoints
            )
        axis.set_title(
            "Divergent restoring mass" if schedule_name == "divergent" else "Finite restoring mass"
        )
        axis.set_xlabel("round")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("MSE / squared initial error")
    axes[0].legend()
    fig.suptitle("Barycentric MLE recovery across state-dependent likelihood families")
    _save_figure(fig, output / "figure_theory_family_coverage")
    return pd.DataFrame(rows)


def run_constant_rate_transition(output: Path, profile: str) -> pd.DataFrame:
    max_round = 20_000 if profile == "full" else 1200
    paths = 3500 if profile == "full" else 800
    checkpoints = np.unique(np.geomspace(1, max_round, 70).astype(int))
    rows: list[dict[str, object]] = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for index, rho in enumerate([0.25, 0.5, 0.75]):
        real_per_round = int(40 * rho)
        real = [real_per_round] * max_round
        synthetic = [40 - real_per_round] * max_round
        exact = gaussian_exact_mse(real, synthetic)
        simulated = simulate_barycentric_mse(
            "gaussian", real, synthetic, paths, PAPER_SEED + index
        )
        axes[0].loglog(np.arange(1, max_round + 1), exact, label=f"rho={rho:g} exact")
        axes[0].scatter(checkpoints, simulated[checkpoints - 1], s=13, facecolors="none")
        if rho < 0.5:
            normalized = np.power(np.arange(1, max_round + 1), 2 * rho) * exact
        elif rho == 0.5:
            t = np.arange(1, max_round + 1)
            normalized = t * exact / np.log(np.maximum(t, 2))
        else:
            normalized = np.arange(1, max_round + 1) * exact
        axes[1].semilogx(np.arange(1, max_round + 1), normalized, label=f"rho={rho:g}")
        rows.extend(
            {
                "experiment": "constant_rate_transition",
                "rho": rho,
                "round": int(t),
                "exact_mse": float(exact[t - 1]),
                "monte_carlo_mse": float(simulated[t - 1]),
                "paths": paths,
            }
            for t in checkpoints
        )
    axes[0].set(xlabel="round", ylabel="MSE", title="Exact law and Monte Carlo")
    axes[1].set(xlabel="round", ylabel="theorem-normalized MSE", title="Transition at rho = 1/2")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    _save_figure(fig, output / "figure_theory_rate_transition")
    return pd.DataFrame(rows)


def _power_log_counts(max_round: int, r: float) -> tuple[list[int], list[int]]:
    t = np.arange(2, max_round + 2, dtype=float)
    logs = np.log(t)
    synthetic = np.maximum(1, np.rint(40 * logs**2)).astype(int)
    real = np.maximum(1, np.rint(20 * logs ** (2 - r))).astype(int)
    return real.tolist(), synthetic.tolist()


def run_power_log_boundary(output: Path, profile: str) -> pd.DataFrame:
    exact_rounds = 300_000 if profile == "full" else 6000
    monte_carlo_rounds = 12_000 if profile == "full" else 800
    paths = 2000 if profile == "full" else 500
    checkpoints = np.unique(np.geomspace(1, monte_carlo_rounds, 45).astype(int))
    rows: list[dict[str, object]] = []
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for index, (axis, r) in enumerate(zip(axes, [0.5, 1.0, 1.5], strict=True)):
        real, synthetic = _power_log_counts(exact_rounds, r)
        exact = gaussian_exact_mse(real, synthetic)
        mc_real, mc_synthetic = real[:monte_carlo_rounds], synthetic[:monte_carlo_rounds]
        simulated = simulate_barycentric_mse(
            "gaussian", mc_real, mc_synthetic, paths, PAPER_SEED + index
        )
        axis.loglog(np.arange(1, exact_rounds + 1), exact, label="exact recursion")
        axis.scatter(checkpoints, simulated[checkpoints - 1], s=11, facecolors="none", label="Monte Carlo")
        axis.set_title(f"r = {r:g}")
        axis.set_xlabel("round")
        axis.grid(alpha=0.25)
        rows.extend(
            {
                "experiment": "power_log_boundary",
                "r": r,
                "round": int(t),
                "exact_mse": float(exact[t - 1]),
                "monte_carlo_mse": float(simulated[t - 1]),
                "paths": paths,
            }
            for t in checkpoints
        )
    axes[0].set_ylabel("MSE")
    axes[0].legend()
    fig.suptitle("Power-log schedule boundary")
    _save_figure(fig, output / "figure_theory_power_log")
    return pd.DataFrame(rows)


def _sample_u_star(paths: int, rng: np.random.Generator, max_steps: int = 300) -> np.ndarray:
    exponentials = rng.exponential(size=(paths, max_steps + 1))
    tau = np.cumsum(exponentials, axis=1)
    k = np.arange(max_steps + 1)
    scores = tau / 4.0 - k * math.log(2.0)
    argmax = np.argmax(scores, axis=1)
    return tau[np.arange(paths), argmax] / 2.0


def _finite_first_fit_alpha(
    sample_count: int,
    paths: int,
    rng: np.random.Generator,
    max_upper_order: int = 100,
) -> np.ndarray:
    """Sample the finite-n first MLE over the relevant upper-order breakpoints.

    The top order statistics are sampled jointly without allocating the complete
    sample matrix. Lemma 40 controls escape below this upper window; the retained
    window size is recorded in every output row so this numerical truncation is
    explicit rather than presented as a theorem assumption.
    """

    upper = min(max_upper_order, sample_count - 1)
    if upper <= 0:
        return np.zeros(paths)
    threshold = rng.beta(sample_count - upper, upper + 1, size=paths)
    conditional = np.sort(rng.random((paths, upper)), axis=1)
    top = threshold[:, None] + (1.0 - threshold[:, None]) * conditional
    order_statistics = np.concatenate([threshold[:, None], top], axis=1)
    k = np.arange(upper + 1)
    values = order_statistics[:, upper - k]
    alpha = (1.0 - values) / 2.0
    valid = alpha <= 0.25
    h = (2.0 - 3.0 * alpha) / (2.0 * (1.0 - 2.0 * alpha))
    likelihood = (sample_count - k[None, :]) * np.log(h) - k[None, :] * math.log(2)
    likelihood[~valid] = -np.inf
    best_index = np.argmax(
        np.concatenate([np.zeros((paths, 1)), likelihood], axis=1), axis=1
    )
    selected = np.zeros(paths)
    nonzero = best_index > 0
    selected[nonzero] = alpha[np.arange(paths)[nonzero], best_index[nonzero] - 1]
    return selected


def _critical_window_probability(n: int, tau: np.ndarray, paths: int, rng: np.random.Generator) -> np.ndarray:
    d_star = -(0.8 * math.log(17 / 16) + 0.2 * math.log(0.5))
    c_crit = 2 * math.log(128) / d_star - 2
    output = np.empty_like(tau, dtype=float)
    for index, value in enumerate(tau):
        count = max(1, int(round((2 + c_crit + value / math.sqrt(n)) * n)))
        low_count = rng.binomial(count, 0.8, size=paths)
        bulk = low_count * math.log(17 / 16) + (count - low_count) * math.log(0.5)
        output[index] = np.mean(2 * n * math.log(128) + bulk > 0)
    return output


def run_spike_threshold(output: Path, profile: str) -> pd.DataFrame:
    u_paths = 180_000 if profile == "full" else 15_000
    finite_paths = 90_000 if profile == "full" else 5_000
    critical_paths = 90_000 if profile == "full" else 8_000
    rng = np.random.default_rng(PAPER_SEED)
    u_star = _sample_u_star(u_paths, rng)
    d_star = -(0.8 * math.log(17 / 16) + 0.2 * math.log(0.5))
    c_crit = 2 * math.log(128) / d_star - 2
    g_values = np.array([math.log(17 / 16), math.log(0.5)])
    sigma_star = math.sqrt((2 + c_crit) * np.var(np.repeat(g_values, [8, 2])))
    x_grid = np.linspace(0, 15, 61)
    pi = np.mean(
        1.0 - np.exp(-u_star[:, None] / (2.0 * (1.0 + x_grid[None, :]))), axis=0
    )
    tau_grid = np.linspace(-120, 120, 49)
    n_values = [50, 200, 800]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].plot(x_grid, pi, label="limit pi(x)")
    axes[0].set(xlabel="x = m0/n", ylabel="spike probability", title="Spike production")
    rows: list[dict[str, object]] = [
        {
            "experiment": "spike_threshold",
            "panel": "spike_production",
            "x": float(x),
            "probability": float(probability),
            "paths": u_paths,
            "c_crit": c_crit,
        }
        for x, probability in zip(x_grid, pi, strict=True)
    ]
    finite_x = np.linspace(0, 15, 31)
    for n in [80, 320]:
        finite_probabilities: list[float] = []
        for x in finite_x:
            sample_count = n + int(round(x * n))
            alpha = _finite_first_fit_alpha(sample_count, finite_paths, rng)
            probability = float(np.mean(1.0 - np.power(1.0 - alpha / 2.0, n)))
            finite_probabilities.append(probability)
            rows.append(
                {
                    "experiment": "spike_threshold",
                    "panel": "finite_spike_production",
                    "n": n,
                    "x": float(x),
                    "probability": probability,
                    "paths": finite_paths,
                    "upper_order_window": 100,
                    "c_crit": c_crit,
                }
            )
        axes[0].plot(
            finite_x,
            finite_probabilities,
            linestyle="--",
            linewidth=1.2,
            label=f"finite n={n}",
        )
    gaussian_limit = norm.cdf(-d_star * tau_grid / sigma_star)
    axes[1].plot(tau_grid, gaussian_limit, color="black", label="Gaussian limit")
    for n in n_values:
        simulated = _critical_window_probability(n, tau_grid, critical_paths, rng)
        axes[1].plot(tau_grid, simulated, marker="o", markersize=2, linewidth=1, label=f"n={n}")
        rows.extend(
            {
                "experiment": "spike_threshold",
                "panel": "critical_window",
                "n": n,
                "tau": float(tau),
                "probability": float(probability),
                "gaussian_limit": float(limit),
                "paths": critical_paths,
                "c_crit": c_crit,
            }
            for tau, probability, limit in zip(tau_grid, simulated, gaussian_limit, strict=True)
        )
    axes[1].set(xlabel="tau", ylabel="spike-win probability", title="n^-1/2 critical window")
    surface_x = np.linspace(0, 15, 80)
    surface_y = np.linspace(85, 115, 100)
    surface_pi = np.interp(surface_x, x_grid, pi)
    total_offset = np.sqrt(400) * (surface_x[None, :] + surface_y[:, None] - c_crit)
    surface = surface_pi[None, :] * norm.cdf(-d_star * total_offset / sigma_star)
    image = axes[2].imshow(
        surface,
        origin="lower",
        aspect="auto",
        extent=[surface_x.min(), surface_x.max(), surface_y.min(), surface_y.max()],
        cmap="viridis",
    )
    axes[2].plot(surface_x, c_crit - surface_x, color="white", linewidth=1.5)
    axes[2].set(xlabel="m0/n", ylabel="m1/n", title="Predicted collapse, n=400")
    fig.colorbar(image, ax=axes[2], label="probability")
    for axis in axes[:2]:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle(f"Construction 1: ccrit = {c_crit:.3f}, simulated E[U*] = {u_star.mean():.3f}")
    _save_figure(fig, output / "figure_theory_spike_threshold")
    return pd.DataFrame(rows)


def run_theory_suite(output_dir: str | Path, profile: str = "quick") -> dict[str, object]:
    if profile not in {"quick", "full"}:
        raise ValueError("profile must be quick or full")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frames = [
        run_family_coverage(output, profile),
        run_constant_rate_transition(output, profile),
        run_power_log_boundary(output, profile),
        run_spike_threshold(output, profile),
    ]
    results = pd.concat(frames, ignore_index=True, sort=False)
    results.to_csv(output / "theory_results.csv", index=False)
    family = results[results["experiment"] == "family_coverage"].dropna(
        subset=["family"]
    )
    family_final = family.sort_values("round").groupby(["schedule", "family"]).tail(1)
    constant = results[results["experiment"] == "constant_rate_transition"].dropna(
        subset=["exact_mse"]
    ).copy()
    constant["relative_error"] = (
        (constant["monte_carlo_mse"] - constant["exact_mse"]).abs()
        / constant["exact_mse"]
    )
    power_log = results[results["experiment"] == "power_log_boundary"].dropna(
        subset=["exact_mse"]
    ).copy()
    power_log["relative_error"] = (
        (power_log["monte_carlo_mse"] - power_log["exact_mse"]).abs()
        / power_log["exact_mse"]
    )
    critical = results[results.get("panel", pd.Series(index=results.index, dtype=object)) == "critical_window"].copy()
    if not critical.empty:
        critical["absolute_error"] = (
            critical["probability"] - critical["gaussian_limit"]
        ).abs()
    manifest = {
        "profile": profile,
        "seed": PAPER_SEED,
        "experiments": sorted(results["experiment"].unique().tolist()),
        "rows": int(len(results)),
        "checks": {
            "family_final_normalized_mse": family_final[
                ["schedule", "family", "normalized_mse"]
            ].to_dict(orient="records"),
            "constant_rate_max_relative_mc_error": {
                str(key): float(value)
                for key, value in constant.groupby("rho")["relative_error"].max().items()
            },
            "power_log_max_relative_mc_error": {
                str(key): float(value)
                for key, value in power_log.groupby("r")["relative_error"].max().items()
            },
            "critical_window_max_absolute_error": {
                str(key): float(value)
                for key, value in critical.groupby("n")["absolute_error"].max().items()
            },
        },
        "files": sorted(path.name for path in output.iterdir()),
        "interpretation": (
            "These Monte Carlo experiments check theorem predictions; they are not logical "
            "evidence for unproved extensions to neural language models."
        ),
    }
    atomic_write_json(output / "manifest.json", manifest)
    return manifest
