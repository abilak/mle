from __future__ import annotations

import copy
import gc
import gzip
import hashlib
import math
import os
import shutil
import struct
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .generative_data import exclusive_file_lock
from .io import atomic_write_json, read_json, seed_everything, stable_hash


def _dependencies() -> tuple[Any, Any]:
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError(
            "Install the generative extra (torch) for vision experiments"
        ) from exc
    return torch, nn


def _device() -> str:
    torch, _ = _dependencies()
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _clear() -> None:
    torch, _ = _dependencies()
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


_VISION_DATASETS = {
    "MNIST": {
        "base_url": "https://storage.googleapis.com/cvdf-datasets/mnist",
        "files": {
            "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
            "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
            "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
            "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
        },
    },
    "FashionMNIST": {
        "base_url": (
            "https://raw.githubusercontent.com/zalandoresearch/"
            "fashion-mnist/master/data/fashion"
        ),
        "files": {
            "train-images-idx3-ubyte.gz": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
            "train-labels-idx1-ubyte.gz": "25c81989df183df01b3e8a0aad5dffbe",
            "t10k-images-idx3-ubyte.gz": "bef4ecab320f06d8554ea6380940ec79",
            "t10k-labels-idx1-ubyte.gz": "bb300cfdad3c16e7a12a480ee83cd310",
        },
    },
}


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_idx_file(url: str, destination: Path, expected_md5: str) -> None:
    if destination.is_file() and _md5(destination) == expected_md5:
        return
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("requests is required to download the vision datasets") from exc

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.part")
    temporary.unlink(missing_ok=True)
    try:
        with requests.get(url, stream=True, timeout=(15, 180)) as response:
            response.raise_for_status()
            with temporary.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
        actual_md5 = _md5(temporary)
        if actual_md5 != expected_md5:
            raise RuntimeError(
                f"Checksum mismatch for {url}: expected {expected_md5}, got {actual_md5}"
            )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        magic, count, rows, columns = struct.unpack(">IIII", handle.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid IDX image magic number in {path}: {magic}")
        values = np.frombuffer(handle.read(), dtype=np.uint8)
    expected = count * rows * columns
    if values.size != expected:
        raise ValueError(f"Truncated IDX image file {path}: expected {expected} bytes")
    return values.reshape(count, 1, rows, columns)


def _read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        magic, count = struct.unpack(">II", handle.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid IDX label magic number in {path}: {magic}")
        values = np.frombuffer(handle.read(), dtype=np.uint8)
    if values.size != count:
        raise ValueError(f"Truncated IDX label file {path}: expected {count} bytes")
    return values.astype(np.int64, copy=False)


def load_mnist_arrays(
    config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dataset_name = str(config.get("vision_dataset", "MNIST"))
    if dataset_name not in _VISION_DATASETS:
        raise ValueError("vision_dataset must be MNIST or FashionMNIST")
    metadata = _VISION_DATASETS[dataset_name]
    root = Path(str(config.get("vision_data_dir", "data/generative/vision")))
    raw = root / dataset_name / "raw"
    with exclusive_file_lock(root / f".{dataset_name}.lock"):
        for filename, expected_md5 in metadata["files"].items():
            _download_idx_file(
                f"{metadata['base_url']}/{filename}", raw / filename, expected_md5
            )
    train_images = _read_idx_images(raw / "train-images-idx3-ubyte.gz").astype(
        np.float32
    ) / 255.0
    test_images = _read_idx_images(raw / "t10k-images-idx3-ubyte.gz").astype(
        np.float32
    ) / 255.0
    train_labels = _read_idx_labels(raw / "train-labels-idx1-ubyte.gz")
    test_labels = _read_idx_labels(raw / "t10k-labels-idx1-ubyte.gz")
    if len(train_images) != len(train_labels) or len(test_images) != len(test_labels):
        raise ValueError(f"Image/label count mismatch in the downloaded {dataset_name} data")
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


def flatten_image_batch(values: np.ndarray) -> np.ndarray:
    """Flatten a batch while preserving a valid feature width for empty batches."""

    array = np.asarray(values)
    if array.ndim < 2:
        raise ValueError("An image batch must include a batch and feature dimension")
    feature_count = int(np.prod(array.shape[1:]))
    if feature_count <= 0:
        raise ValueError("An image batch must have at least one feature")
    return array.reshape((len(array), feature_count))


def preprocess_flow_images(
    images: np.ndarray,
    specification: Mapping[str, Any],
    seed: int,
) -> np.ndarray:
    """Prepare real images for a flow, with optional uniform dequantization."""

    values = np.asarray(images, dtype=np.float32)
    preprocessing = dict(specification.get("flow_preprocessing", {}))
    alpha = float(preprocessing.get("logit_alpha", 1e-4))
    if not 0 < alpha < 0.5:
        raise ValueError("vision.flow_preprocessing.logit_alpha must lie in (0, 0.5)")
    if bool(preprocessing.get("dequantize", False)):
        rng = np.random.default_rng(seed)
        pixels = np.clip(np.rint(values * 255), 0, 255)
        values = (pixels + rng.random(values.shape, dtype=np.float32)) / 256.0
    return flatten_image_batch(logit_transform(values, alpha=alpha))


def inverse_logit(values: np.ndarray, alpha: float = 0.0) -> np.ndarray:
    if not 0 <= alpha < 0.5:
        raise ValueError("Logit transform alpha must lie in [0, 0.5)")
    clipped = np.clip(values, -20, 20)
    expected = 28 * 28
    flat = flatten_image_batch(clipped)
    if flat.shape[1] != expected:
        raise ValueError(f"Expected {expected} flow features, received {flat.shape[1]}")
    probabilities = 1 / (1 + np.exp(-flat))
    pixels = np.clip((probabilities - alpha) / (1 - 2 * alpha), 0, 1)
    return pixels.reshape((len(flat), 1, 28, 28)).astype(np.float32)


def _flow_classes() -> tuple[type, type, type, type]:
    torch, nn = _dependencies()

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

    class ConvolutionalCoupling(nn.Module):
        def __init__(
            self,
            image_shape: tuple[int, int, int],
            hidden: int,
            mask: Any,
        ):
            super().__init__()
            channels, _, _ = image_shape
            self.image_shape = image_shape
            self.register_buffer("mask", mask)
            self.network = nn.Sequential(
                nn.Conv2d(channels, hidden, 3, padding=1),
                nn.SiLU(),
                nn.Conv2d(hidden, hidden, 1),
                nn.SiLU(),
                nn.Conv2d(hidden, 2 * channels, 3, padding=1),
            )
            nn.init.zeros_(self.network[-1].weight)
            nn.init.zeros_(self.network[-1].bias)

        def _image(self, value: Any) -> Any:
            return value.reshape((len(value),) + self.image_shape)

        def forward(self, value: Any) -> tuple[Any, Any]:
            image = self._image(value)
            fixed = image * self.mask
            scale, shift = self.network(fixed).chunk(2, dim=1)
            scale = 1.8 * torch.tanh(scale / 1.8) * (1 - self.mask)
            shift = shift * (1 - self.mask)
            transformed = fixed + (1 - self.mask) * (
                image * torch.exp(scale) + shift
            )
            return transformed.flatten(1), scale.flatten(1).sum(dim=-1)

        def inverse(self, value: Any) -> Any:
            image = self._image(value)
            fixed = image * self.mask
            scale, shift = self.network(fixed).chunk(2, dim=1)
            scale = 1.8 * torch.tanh(scale / 1.8) * (1 - self.mask)
            shift = shift * (1 - self.mask)
            transformed = fixed + (1 - self.mask) * (image - shift) * torch.exp(
                -scale
            )
            return transformed.flatten(1)

    class ImageRealNVP(nn.Module):
        def __init__(
            self,
            image_shape: tuple[int, int, int] = (1, 28, 28),
            hidden: int = 32,
            layers: int = 8,
        ):
            super().__init__()
            channels, height, width = image_shape
            dimension = channels * height * width
            checkerboard = (
                torch.arange(height)[:, None] + torch.arange(width)[None, :]
            ) % 2
            base_mask = checkerboard.float()[None, None]
            if channels > 1:
                base_mask = base_mask.repeat(1, channels, 1, 1)
            self.layers = nn.ModuleList(
                ConvolutionalCoupling(
                    image_shape,
                    hidden,
                    base_mask if index % 2 == 0 else 1 - base_mask,
                )
                for index in range(layers)
            )
            self.dimension = dimension
            self.specification = {
                "architecture": "conv",
                "dimension": dimension,
                "image_shape": list(image_shape),
                "coupling_channels": hidden,
                "layers": layers,
            }

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

    return Coupling, RealNVP, ConvolutionalCoupling, ImageRealNVP


def _make_flow(specification: Mapping[str, Any]) -> Any:
    _, RealNVP, _, ImageRealNVP = _flow_classes()
    architecture = str(specification.get("architecture", "mlp"))
    if architecture == "mlp":
        return RealNVP(
            dimension=int(specification.get("dimension", 784)),
            hidden=int(specification.get("hidden", 256)),
            layers=int(specification.get("layers", 6)),
        )
    if architecture == "conv":
        image_shape = tuple(
            int(value) for value in specification.get("image_shape", (1, 28, 28))
        )
        if len(image_shape) != 3 or any(value <= 0 for value in image_shape):
            raise ValueError("vision.flow_model.image_shape must contain three positive values")
        dimension = math.prod(image_shape)
        expected_dimension = int(specification.get("dimension", dimension))
        if dimension != expected_dimension:
            raise ValueError("Flow dimension does not match vision.flow_model.image_shape")
        hidden = int(specification.get("coupling_channels", 32))
        layers = int(specification.get("layers", 8))
        if hidden <= 0 or layers <= 0:
            raise ValueError("Flow convolution channels and layers must be positive")
        return ImageRealNVP(
            image_shape=image_shape,
            hidden=hidden,
            layers=layers,
        )
    raise ValueError("vision.flow_model.architecture must be mlp or conv")


def save_flow(model: Any, target: str | Path, summary: Mapping[str, Any]) -> None:
    torch, _ = _dependencies()
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
    torch, _ = _dependencies()
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
    torch, _ = _dependencies()
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
    flat = flatten_image_batch(transformed_images)
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
    torch, _ = _dependencies()
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
    flat = flatten_image_batch(values)
    if not len(flat):
        return np.empty(0, dtype=np.float64)
    torch, _ = _dependencies()
    device = _device()
    model = load_flow(checkpoint).to(device).eval()
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
    _, nn = _dependencies()

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
    torch, _ = _dependencies()
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
    torch, _ = _dependencies()
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
    target_probability = float(uniform[missing_class])
    missing_class_probability = float(probabilities[missing_class])
    return {
        "class_probabilities": probabilities.tolist(),
        "class_kl_to_uniform": kl,
        "missing_class_probability": missing_class_probability,
        "missing_class_target_probability": target_probability,
        "missing_class_absolute_error": abs(
            missing_class_probability - target_probability
        ),
        "class_entropy": float(-np.sum(probabilities * np.log(probabilities + 1e-12))),
    }


def frechet_feature_distance(reference: np.ndarray, candidate: np.ndarray) -> float:
    import warnings

    from scipy.linalg import LinAlgWarning, sqrtm

    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if reference.ndim != 2 or candidate.ndim != 2:
        raise ValueError("Feature arrays must be two-dimensional")
    if reference.shape[1] != candidate.shape[1]:
        raise ValueError("Reference and candidate features must have equal widths")
    if len(reference) < 2 or len(candidate) < 2:
        raise ValueError("Fréchet distance requires at least two samples per set")

    reference_mean, candidate_mean = reference.mean(axis=0), candidate.mean(axis=0)
    reference_cov = np.atleast_2d(np.cov(reference, rowvar=False))
    candidate_cov = np.atleast_2d(np.cov(candidate, rowvar=False))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LinAlgWarning)
        covariance_root = sqrtm(reference_cov @ candidate_cov)
    if not np.isfinite(covariance_root).all():
        dimension = reference_cov.shape[0]
        scale = max(
            float(np.trace(reference_cov)) / dimension,
            float(np.trace(candidate_cov)) / dimension,
            1.0,
        )
        jitter = np.eye(dimension) * (1e-6 * scale)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", LinAlgWarning)
            covariance_root = sqrtm(
                (reference_cov + jitter) @ (candidate_cov + jitter)
            )
    if np.iscomplexobj(covariance_root):
        covariance_root = covariance_root.real
    difference = reference_mean - candidate_mean
    distance = float(
        difference @ difference
        + np.trace(reference_cov + candidate_cov - 2 * covariance_root)
    )
    return max(0.0, distance)


def _diffusion_classes() -> tuple[type, type]:
    torch, nn = _dependencies()

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

    class ResidualBlock(nn.Module):
        def __init__(self, channels: int, time_dim: int):
            super().__init__()
            groups = min(8, channels)
            while channels % groups:
                groups -= 1
            self.norm1 = nn.GroupNorm(groups, channels)
            self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
            self.time_projection = nn.Linear(time_dim, channels)
            self.norm2 = nn.GroupNorm(groups, channels)
            self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

        def forward(self, value: Any, time_embedding: Any) -> Any:
            hidden = self.conv1(torch.nn.functional.silu(self.norm1(value)))
            hidden = hidden + self.time_projection(time_embedding)[:, :, None, None]
            hidden = self.conv2(torch.nn.functional.silu(self.norm2(hidden)))
            return value + hidden

    class ConvDenoiser(nn.Module):
        def __init__(self, channels: int = 64, time_dim: int = 64, depth: int = 8):
            super().__init__()
            self.channels = channels
            self.time_dim = time_dim
            self.depth = depth
            self.input = nn.Conv2d(1, channels, 3, padding=1)
            self.time_network = nn.Sequential(
                nn.Linear(time_dim, 4 * time_dim),
                nn.SiLU(),
                nn.Linear(4 * time_dim, time_dim),
            )
            self.blocks = nn.ModuleList(
                ResidualBlock(channels, time_dim) for _ in range(depth)
            )
            groups = min(8, channels)
            while channels % groups:
                groups -= 1
            self.output_norm = nn.GroupNorm(groups, channels)
            self.output = nn.Conv2d(channels, 1, 3, padding=1)
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

        def time_embedding(self, timesteps: Any, total_steps: int) -> Any:
            half = self.time_dim // 2
            frequencies = torch.exp(
                -math.log(10_000)
                * torch.arange(half, device=timesteps.device)
                / max(1, half - 1)
            )
            scaled = timesteps.float() / max(1, total_steps - 1)
            angles = scaled[:, None] * frequencies[None, :] * 1000
            embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
            return self.time_network(embedding)

        def forward(self, images: Any, timesteps: Any, total_steps: int) -> Any:
            time_embedding = self.time_embedding(timesteps, total_steps)
            hidden = self.input(images)
            for block in self.blocks:
                hidden = block(hidden, time_embedding)
            hidden = torch.nn.functional.silu(self.output_norm(hidden))
            return self.output(hidden)

    return TinyDenoiser, ConvDenoiser


def _make_diffusion(specification: Mapping[str, Any]) -> Any:
    TinyDenoiser, ConvDenoiser = _diffusion_classes()
    architecture = str(specification.get("architecture", "mlp"))
    time_dim = int(specification.get("time_dim", 64))
    if time_dim <= 0 or time_dim % 2:
        raise ValueError("vision.diffusion_model.time_dim must be a positive even integer")
    if architecture == "mlp":
        return TinyDenoiser(
            hidden=int(specification.get("hidden", 512)),
            time_dim=time_dim,
        )
    if architecture == "conv":
        channels = int(specification.get("channels", 64))
        depth = int(specification.get("depth", 8))
        if channels <= 0 or depth <= 0:
            raise ValueError("Diffusion convolution channels and depth must be positive")
        return ConvDenoiser(
            channels=channels,
            time_dim=time_dim,
            depth=depth,
        )
    raise ValueError("vision.diffusion_model.architecture must be mlp or conv")


def diffusion_beta_values(
    steps: int,
    schedule: str = "linear",
    cosine_s: float = 0.008,
) -> np.ndarray:
    """Return the discrete beta schedule used by training and sampling."""

    if steps <= 0:
        raise ValueError("Diffusion steps must be positive")
    if schedule == "linear":
        return np.linspace(1e-4, 0.02, steps, dtype=np.float64)
    if schedule == "cosine":
        if cosine_s < 0:
            raise ValueError("Cosine schedule offset must be nonnegative")
        times = np.linspace(0, steps, steps + 1, dtype=np.float64) / steps
        cumulative = np.cos(
            ((times + cosine_s) / (1 + cosine_s)) * math.pi / 2
        ) ** 2
        cumulative = cumulative / cumulative[0]
        betas = 1 - cumulative[1:] / cumulative[:-1]
        return np.clip(betas, 1e-5, 0.999)
    raise ValueError("Diffusion noise schedule must be linear or cosine")


def _diffusion_schedule(
    steps: int,
    device: str,
    schedule: str = "linear",
    cosine_s: float = 0.008,
) -> tuple[Any, Any, Any]:
    torch, _ = _dependencies()
    betas = torch.as_tensor(
        diffusion_beta_values(steps, schedule=schedule, cosine_s=cosine_s),
        dtype=torch.float32,
        device=device,
    )
    alphas = 1 - betas
    cumulative = torch.cumprod(alphas, dim=0)
    return betas, alphas, cumulative


def save_diffusion(model: Any, target: str | Path, summary: Mapping[str, Any]) -> None:
    torch, _ = _dependencies()
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
    torch, _ = _dependencies()
    summary = read_json(Path(checkpoint) / "training_summary.json")
    model = _make_diffusion(summary["model"])
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
    torch, _ = _dependencies()
    seed_everything(seed)
    device = _device()
    model = (
        load_diffusion(source_checkpoint)[0]
        if source_checkpoint
        else _make_diffusion(specification)
    )
    model.to(device).train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("learning_rate", 2e-4)),
        weight_decay=float(training.get("weight_decay", 0.01)),
    )
    diffusion_steps = int(specification.get("diffusion_steps", 50))
    noise_schedule = str(specification.get("noise_schedule", "linear"))
    cosine_s = float(specification.get("cosine_s", 0.008))
    _, _, cumulative = _diffusion_schedule(
        diffusion_steps,
        device,
        schedule=noise_schedule,
        cosine_s=cosine_s,
    )
    terminal_signal = float(cumulative[-1].detach().cpu())
    if bool(specification.get("require_near_pure_noise", False)) and terminal_signal > 1e-3:
        raise ValueError(
            "Diffusion schedule leaves too much terminal signal for pure-noise sampling: "
            f"alpha_bar_T={terminal_signal:.6g}"
        )
    rng = np.random.default_rng(seed)
    steps = int(training.get("max_steps", 1000))
    batch_size = int(training.get("batch_size", 128))
    losses = []
    ema_decay = float(training.get("ema_decay", 0.0))
    if not 0 <= ema_decay < 1:
        raise ValueError("Diffusion EMA decay must lie in [0, 1)")
    ema_model = copy.deepcopy(model).eval() if ema_decay else None
    if ema_model is not None:
        for parameter in ema_model.parameters():
            parameter.requires_grad_(False)
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
        if ema_model is not None:
            with torch.no_grad():
                for averaged, current in zip(
                    ema_model.parameters(), model.parameters(), strict=True
                ):
                    averaged.mul_(ema_decay).add_(current, alpha=1 - ema_decay)
        losses.append(float(loss.detach().cpu()))
    architecture = str(specification.get("architecture", "mlp"))
    summary = {
        "seed": seed,
        "examples": int(len(images)),
        "steps": steps,
        "final_loss": losses[-1],
        "mean_tail_loss": float(np.mean(losses[-min(20, len(losses)) :])),
        "model": {
            "architecture": architecture,
            "hidden": int(specification.get("hidden", 512)),
            "channels": int(specification.get("channels", 64)),
            "depth": int(specification.get("depth", 8)),
            "time_dim": int(specification.get("time_dim", 64)),
            "diffusion_steps": diffusion_steps,
            "noise_schedule": noise_schedule,
            "cosine_s": cosine_s,
            "terminal_signal": terminal_signal,
        },
        "ema_decay": ema_decay,
        "device": device,
    }
    save_diffusion(ema_model if ema_model is not None else model, output_dir, summary)
    del model, optimizer, ema_model
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
    torch, _ = _dependencies()
    seed_everything(seed)
    device = _device()
    model, summary = load_diffusion(checkpoint)
    model.to(device).eval()
    steps = int(summary["model"]["diffusion_steps"])
    noise_schedule = str(summary["model"].get("noise_schedule", "linear"))
    cosine_s = float(summary["model"].get("cosine_s", 0.008))
    betas, alphas, cumulative = _diffusion_schedule(
        steps,
        device,
        schedule=noise_schedule,
        cosine_s=cosine_s,
    )
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
