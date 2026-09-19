from __future__ import annotations

import gc
import math
import shutil
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .generative_data import exclusive_file_lock
from .io import atomic_write_json, read_json, seed_everything, stable_hash


def _dependencies() -> tuple[Any, Any, Any]:
    try:
        import torch
        import torch.nn as nn
        import torchvision
    except ImportError as exc:
        raise RuntimeError(
            "Install the generative extra (torch and torchvision) for vision experiments"
        ) from exc
    return torch, nn, torchvision


def _device() -> str:
    torch, _, _ = _dependencies()
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _clear() -> None:
    torch, _, _ = _dependencies()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    mps = getattr(torch.backends, "mps", None)
    if (
        mps is not None
        and mps.is_available()
        and getattr(torch, "mps", None) is not None
        and hasattr(torch.mps, "empty_cache")
    ):
        torch.mps.empty_cache()


def load_mnist_arrays(config: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _, _, torchvision = _dependencies()
    dataset_name = str(config.get("vision_dataset", "MNIST"))
    if dataset_name not in {"MNIST", "FashionMNIST"}:
        raise ValueError("vision_dataset must be MNIST or FashionMNIST")
    dataset_class = getattr(torchvision.datasets, dataset_name)
    root = str(config.get("vision_data_dir", "data/generative/vision"))
    with exclusive_file_lock(Path(root) / f".{dataset_name}.lock"):
        train = dataset_class(root=root, train=True, download=True)
        test = dataset_class(root=root, train=False, download=True)
    train_images = np.asarray(train.data, dtype=np.float32)[:, None] / 255.0
    test_images = np.asarray(test.data, dtype=np.float32)[:, None] / 255.0
    train_labels = np.asarray(train.targets, dtype=np.int64)
    test_labels = np.asarray(test.targets, dtype=np.int64)
    return train_images, train_labels, test_images, test_labels


def balanced_indices(labels: np.ndarray, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    classes = sorted(int(value) for value in np.unique(labels))
    base, remainder = divmod(count, len(classes))
    chosen: list[int] = []
    for position, label in enumerate(classes):
        candidates = np.flatnonzero(labels == label)
        amount = base + int(position < remainder)
        chosen.extend(rng.choice(candidates, size=amount, replace=amount > len(candidates)).tolist())
    rng.shuffle(chosen)
    return np.asarray(chosen, dtype=np.int64)


def logit_transform(images: np.ndarray, alpha: float = 1e-4) -> np.ndarray:
    values = alpha + (1 - 2 * alpha) * np.asarray(images, dtype=np.float32)
    return np.log(values) - np.log1p(-values)


def inverse_logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -20, 20)
    return (1 / (1 + np.exp(-clipped))).reshape((-1, 1, 28, 28)).astype(np.float32)


def _flow_classes() -> tuple[type, type]:
    torch, nn, _ = _dependencies()

    class Coupling(nn.Module):
        def __init__(self, dimension: int, hidden: int, mask: Any):
            super().__init__()
            self.register_buffer("mask", mask)
            self.network = nn.Sequential(
                nn.Linear(dimension, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 2 * dimension),
            )
            nn.init.zeros_(self.network[-1].weight)
            nn.init.zeros_(self.network[-1].bias)

        def forward(self, value: Any) -> tuple[Any, Any]:
            fixed = value * self.mask
            scale, shift = self.network(fixed).chunk(2, dim=-1)
            scale = 1.8 * torch.tanh(scale / 1.8) * (1 - self.mask)
            shift = shift * (1 - self.mask)
            transformed = fixed + (1 - self.mask) * (value * torch.exp(scale) + shift)
            return transformed, scale.sum(dim=-1)

        def inverse(self, value: Any) -> Any:
            fixed = value * self.mask
            scale, shift = self.network(fixed).chunk(2, dim=-1)
            scale = 1.8 * torch.tanh(scale / 1.8) * (1 - self.mask)
            shift = shift * (1 - self.mask)
            return fixed + (1 - self.mask) * (value - shift) * torch.exp(-scale)

    class RealNVP(nn.Module):
        def __init__(self, dimension: int = 784, hidden: int = 256, layers: int = 6):
            super().__init__()
            modules = []
            base_mask = (torch.arange(dimension) % 2).float()
            for index in range(layers):
                modules.append(Coupling(dimension, hidden, base_mask if index % 2 == 0 else 1 - base_mask))
            self.layers = nn.ModuleList(modules)
            self.dimension = dimension
            self.specification = {"dimension": dimension, "hidden": hidden, "layers": layers}

        def log_prob(self, value: Any) -> Any:
            latent = value
            log_det = torch.zeros(value.shape[0], device=value.device)
            for layer in self.layers:
                latent, contribution = layer(latent)
                log_det += contribution
            base = -0.5 * (latent.square() + math.log(2 * math.pi)).sum(dim=-1)
            return base + log_det

        def sample(self, count: int, device: str) -> Any:
            value = torch.randn(count, self.dimension, device=device)
            for layer in reversed(self.layers):
                value = layer.inverse(value)
            return value

    return Coupling, RealNVP


def _make_flow(specification: Mapping[str, Any]) -> Any:
    _, RealNVP = _flow_classes()
    return RealNVP(
        dimension=int(specification.get("dimension", 784)),
        hidden=int(specification.get("hidden", 256)),
        layers=int(specification.get("layers", 6)),
    )


def save_flow(model: Any, target: str | Path, summary: Mapping[str, Any]) -> None:
    torch, _, _ = _dependencies()
    destination = Path(target)
    temporary = destination.with_name(f".{destination.name}.tmp-{stable_hash(summary, 8)}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    torch.save(model.state_dict(), temporary / "model.pt")
    atomic_write_json(temporary / "training_summary.json", dict(summary) | {"specification": model.specification})
    if destination.exists():
        shutil.rmtree(destination)
    temporary.rename(destination)


def load_flow(checkpoint: str | Path) -> Any:
    torch, _, _ = _dependencies()
    summary = read_json(Path(checkpoint) / "training_summary.json")
    model = _make_flow(summary["specification"])
    model.load_state_dict(torch.load(Path(checkpoint) / "model.pt", map_location="cpu", weights_only=True))
    return model


def train_flow(
    *,
    transformed_images: np.ndarray,
    output_dir: str | Path,
    specification: Mapping[str, Any],
    training: Mapping[str, Any],
    seed: int,
    source_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    if len(transformed_images) == 0:
        raise ValueError("Cannot train a flow on no images")
    torch, _, _ = _dependencies()
    seed_everything(seed)
    device = _device()
    model = load_flow(source_checkpoint) if source_checkpoint else _make_flow(specification)
    model.to(device).train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("learning_rate", 2e-4)),
        weight_decay=float(training.get("weight_decay", 1e-5)),
    )
    rng = np.random.default_rng(seed)
    steps = int(training.get("max_steps", 500))
    batch_size = int(training.get("batch_size", 128))
    losses: list[float] = []
    flat = transformed_images.reshape((len(transformed_images), -1))
    for _ in range(steps):
        indices = rng.integers(0, len(flat), size=batch_size)
        batch = torch.as_tensor(np.asarray(flat[indices]), dtype=torch.float32, device=device)
        loss = -model.log_prob(batch).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(training.get("max_grad_norm", 5.0)))
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    summary = {
        "seed": seed,
        "examples": int(len(flat)),
        "steps": steps,
        "final_nll": losses[-1],
        "mean_tail_nll": float(np.mean(losses[-min(20, len(losses)) :])),
        "device": device,
        "source_checkpoint": str(source_checkpoint) if source_checkpoint else None,
    }
    save_flow(model, output_dir, summary)
    del optimizer, model
    _clear()
    return summary


def sample_flow(checkpoint: str | Path, count: int, seed: int, batch_size: int = 256) -> np.ndarray:
    if count == 0:
        summary = read_json(Path(checkpoint) / "training_summary.json")
        dimension = int(summary["specification"]["dimension"])
        return np.empty((0, dimension), dtype=np.float32)
    torch, _, _ = _dependencies()
    seed_everything(seed)
    device = _device()
    model = load_flow(checkpoint).to(device).eval()
    chunks = []
    with torch.inference_mode():
        for start in range(0, count, batch_size):
            chunks.append(model.sample(min(batch_size, count - start), device).cpu().numpy())
    del model
    _clear()
    return np.concatenate(chunks).astype(np.float32, copy=False)


def flow_log_probabilities(
    checkpoint: str | Path, values: np.ndarray, batch_size: int = 256
) -> np.ndarray:
    torch, _, _ = _dependencies()
    device = _device()
    model = load_flow(checkpoint).to(device).eval()
    flat = values.reshape((len(values), -1))
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(flat), batch_size):
            batch = torch.as_tensor(flat[start : start + batch_size], dtype=torch.float32, device=device)
            chunks.append(model.log_prob(batch).cpu().numpy())
    del model
    _clear()
    return np.concatenate(chunks).astype(np.float64, copy=False)


def ensure_flow_teacher(config: Mapping[str, Any], artifact_root: str | Path) -> Path:
    vision = config["vision"]
    identity = {
        "dataset": config["data"].get("vision_dataset", "MNIST"),
        "model": vision["flow_model"],
        "training": vision["teacher_training"],
        "seed": int(vision.get("teacher_seed", 20260919)),
        "examples": int(vision.get("teacher_examples", 50_000)),
    }
    target = Path(artifact_root) / "flow_teachers" / stable_hash(identity, 16)
    with exclusive_file_lock(target.with_suffix(".lock")):
        if (target / "model.pt").exists():
            return target
        images, labels, _, _ = load_mnist_arrays(config["data"])
        indices = balanced_indices(labels, min(identity["examples"], len(images)), identity["seed"])
        transformed = logit_transform(images[indices])
        train_flow(
            transformed_images=transformed,
            output_dir=target,
            specification=vision["flow_model"],
            training=vision["teacher_training"],
            seed=identity["seed"],
        )
        atomic_write_json(target / "artifact_identity.json", identity)
    return target


def _classifier_class() -> type:
    _, nn, _ = _dependencies()

    class MNISTClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            self.embedding = nn.Sequential(nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU())
            self.head = nn.Linear(128, 10)

        def forward(self, value: Any, return_features: bool = False) -> Any:
            features = self.embedding(self.features(value))
            return features if return_features else self.head(features)

    return MNISTClassifier


def ensure_mnist_classifier(config: Mapping[str, Any], artifact_root: str | Path) -> Path:
    torch, _, _ = _dependencies()
    identity = {
        "dataset": config["data"].get("vision_dataset", "MNIST"),
        "seed": int(config["vision"].get("classifier_seed", 20260921)),
        "steps": int(config["vision"].get("classifier_steps", 1000)),
    }
    target = Path(artifact_root) / "classifiers" / stable_hash(identity, 16)
    with exclusive_file_lock(target.with_suffix(".lock")):
        if (target / "model.pt").exists():
            return target
        train_images, train_labels, test_images, test_labels = load_mnist_arrays(config["data"])
        seed_everything(identity["seed"])
        device = _device()
        classifier = _classifier_class()().to(device).train()
        optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
        rng = np.random.default_rng(identity["seed"])
        batch_size = int(config["vision"].get("classifier_batch_size", 128))
        for _ in range(identity["steps"]):
            indices = rng.integers(0, len(train_images), size=batch_size)
            images = torch.as_tensor(train_images[indices], dtype=torch.float32, device=device)
            labels = torch.as_tensor(train_labels[indices], dtype=torch.long, device=device)
            loss = torch.nn.functional.cross_entropy(classifier(images), labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        classifier.eval()
        correct = 0
        with torch.inference_mode():
            for start in range(0, len(test_images), 512):
                images = torch.as_tensor(test_images[start : start + 512], dtype=torch.float32, device=device)
                prediction = classifier(images).argmax(dim=1).cpu().numpy()
                correct += int((prediction == test_labels[start : start + len(prediction)]).sum())
        target.mkdir(parents=True, exist_ok=True)
        torch.save(classifier.state_dict(), target / "model.pt")
        atomic_write_json(
            target / "summary.json",
            identity | {"test_accuracy": correct / len(test_images)},
        )
        del classifier, optimizer
        _clear()
    return target


def classify_images(
    classifier_checkpoint: str | Path,
    images: np.ndarray,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    torch, _, _ = _dependencies()
    device = _device()
    classifier = _classifier_class()().to(device)
    classifier.load_state_dict(
        torch.load(Path(classifier_checkpoint) / "model.pt", map_location=device, weights_only=True)
    )
    classifier.eval()
    predictions, features = [], []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            batch = torch.as_tensor(images[start : start + batch_size], dtype=torch.float32, device=device)
            predictions.append(classifier(batch).argmax(dim=1).cpu().numpy())
            features.append(classifier(batch, return_features=True).cpu().numpy())
    del classifier
    _clear()
    return np.concatenate(predictions), np.concatenate(features)


def class_distribution_metrics(labels: np.ndarray, missing_class: int = 8) -> dict[str, Any]:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=10).astype(np.float64)
    probabilities = counts / max(1, counts.sum())
    uniform = np.full(10, 0.1)
    kl = float(np.sum(uniform * (np.log(uniform) - np.log(probabilities + 1e-12))))
    return {
        "class_probabilities": probabilities.tolist(),
        "class_kl_to_uniform": kl,
        "missing_class_probability": float(probabilities[missing_class]),
        "class_entropy": float(-np.sum(probabilities * np.log(probabilities + 1e-12))),
    }


def frechet_feature_distance(reference: np.ndarray, candidate: np.ndarray) -> float:
    from scipy.linalg import sqrtm

    reference_mean, candidate_mean = reference.mean(axis=0), candidate.mean(axis=0)
    reference_cov = np.cov(reference, rowvar=False)
    candidate_cov = np.cov(candidate, rowvar=False)
    product_root = sqrtm(reference_cov @ candidate_cov)
    if np.iscomplexobj(product_root):
        product_root = product_root.real
    difference = reference_mean - candidate_mean
    return float(
        difference @ difference
        + np.trace(reference_cov + candidate_cov - 2 * product_root)
    )


def _diffusion_class() -> type:
    torch, nn, _ = _dependencies()

    class TinyDenoiser(nn.Module):
        def __init__(self, hidden: int = 512, time_dim: int = 64):
            super().__init__()
            self.hidden = hidden
            self.time_dim = time_dim
            self.network = nn.Sequential(
                nn.Linear(784 + time_dim, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
                nn.SiLU(),
                nn.Linear(hidden, 784),
            )

        def time_embedding(self, timesteps: Any, total_steps: int) -> Any:
            half = self.time_dim // 2
            frequencies = torch.exp(
                -math.log(10_000) * torch.arange(half, device=timesteps.device) / max(1, half - 1)
            )
            angles = timesteps.float()[:, None] * frequencies[None, :]
            return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)

        def forward(self, images: Any, timesteps: Any, total_steps: int) -> Any:
            flat = images.reshape((len(images), -1))
            embedding = self.time_embedding(timesteps, total_steps)
            return self.network(torch.cat([flat, embedding], dim=1)).reshape_as(images)

    return TinyDenoiser


def _diffusion_schedule(steps: int, device: str) -> tuple[Any, Any, Any]:
    torch, _, _ = _dependencies()
    betas = torch.linspace(1e-4, 0.02, steps, device=device)
    alphas = 1 - betas
    cumulative = torch.cumprod(alphas, dim=0)
    return betas, alphas, cumulative


def save_diffusion(model: Any, target: str | Path, summary: Mapping[str, Any]) -> None:
    torch, _, _ = _dependencies()
    destination = Path(target)
    temporary = destination.with_name(f".{destination.name}.tmp-{stable_hash(summary, 8)}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    torch.save(model.state_dict(), temporary / "model.pt")
    atomic_write_json(temporary / "training_summary.json", dict(summary))
    if destination.exists():
        shutil.rmtree(destination)
    temporary.rename(destination)


def load_diffusion(checkpoint: str | Path) -> tuple[Any, dict[str, Any]]:
    torch, _, _ = _dependencies()
    summary = read_json(Path(checkpoint) / "training_summary.json")
    model = _diffusion_class()(
        hidden=int(summary["model"]["hidden"]), time_dim=int(summary["model"]["time_dim"])
    )
    model.load_state_dict(torch.load(Path(checkpoint) / "model.pt", map_location="cpu", weights_only=True))
    return model, summary


def train_diffusion(
    *,
    images: np.ndarray,
    output_dir: str | Path,
    specification: Mapping[str, Any],
    training: Mapping[str, Any],
    seed: int,
    source_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    if len(images) == 0:
        raise ValueError("Cannot train diffusion on no images")
    torch, _, _ = _dependencies()
    seed_everything(seed)
    device = _device()
    model = (
        load_diffusion(source_checkpoint)[0]
        if source_checkpoint
        else _diffusion_class()(
            hidden=int(specification.get("hidden", 512)),
            time_dim=int(specification.get("time_dim", 64)),
        )
    )
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training.get("learning_rate", 2e-4)))
    diffusion_steps = int(specification.get("diffusion_steps", 50))
    _, _, cumulative = _diffusion_schedule(diffusion_steps, device)
    rng = np.random.default_rng(seed)
    steps = int(training.get("max_steps", 1000))
    batch_size = int(training.get("batch_size", 128))
    losses = []
    normalized = np.asarray(images, dtype=np.float32) * 2 - 1
    for _ in range(steps):
        indices = rng.integers(0, len(normalized), size=batch_size)
        clean = torch.as_tensor(normalized[indices], dtype=torch.float32, device=device)
        timesteps = torch.randint(0, diffusion_steps, (batch_size,), device=device)
        noise = torch.randn_like(clean)
        alpha_bar = cumulative[timesteps].reshape((-1, 1, 1, 1))
        noisy = alpha_bar.sqrt() * clean + (1 - alpha_bar).sqrt() * noise
        prediction = model(noisy, timesteps, diffusion_steps)
        loss = torch.nn.functional.mse_loss(prediction, noise)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    summary = {
        "seed": seed,
        "examples": int(len(images)),
        "steps": steps,
        "final_loss": losses[-1],
        "mean_tail_loss": float(np.mean(losses[-min(20, len(losses)) :])),
        "model": {
            "hidden": int(specification.get("hidden", 512)),
            "time_dim": int(specification.get("time_dim", 64)),
            "diffusion_steps": diffusion_steps,
        },
        "device": device,
    }
    save_diffusion(model, output_dir, summary)
    del model, optimizer
    _clear()
    return summary


def sample_diffusion(
    checkpoint: str | Path,
    count: int,
    seed: int,
    batch_size: int = 256,
) -> np.ndarray:
    if count == 0:
        return np.empty((0, 1, 28, 28), dtype=np.float32)
    torch, _, _ = _dependencies()
    seed_everything(seed)
    device = _device()
    model, summary = load_diffusion(checkpoint)
    model.to(device).eval()
    steps = int(summary["model"]["diffusion_steps"])
    betas, alphas, cumulative = _diffusion_schedule(steps, device)
    results = []
    with torch.inference_mode():
        for start in range(0, count, batch_size):
            current = min(batch_size, count - start)
            value = torch.randn(current, 1, 28, 28, device=device)
            for time_index in reversed(range(steps)):
                time = torch.full((current,), time_index, dtype=torch.long, device=device)
                noise_prediction = model(value, time, steps)
                alpha = alphas[time_index]
                alpha_bar = cumulative[time_index]
                mean = (value - (1 - alpha) / (1 - alpha_bar).sqrt() * noise_prediction) / alpha.sqrt()
                if time_index:
                    previous_alpha_bar = cumulative[time_index - 1]
                    posterior_variance = (
                        betas[time_index]
                        * (1 - previous_alpha_bar)
                        / (1 - alpha_bar)
                    )
                    value = mean + posterior_variance.sqrt() * torch.randn_like(value)
                else:
                    value = mean
            results.append(((value.clamp(-1, 1) + 1) / 2).cpu().numpy())
    del model
    _clear()
    return np.concatenate(results).astype(np.float32, copy=False)
