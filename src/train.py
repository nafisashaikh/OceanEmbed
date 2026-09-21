"""Train the OceanEmbed surface-to-depth reconstruction model."""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import random
import sys
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data.dataset import OceanEmbedDataset, load_normalization_stats
from src.models.reconstruction_model import OceanEmbedReconstructionModel


def _depth_weights(config: dict[str, Any], stats: dict[str, Any]) -> torch.Tensor:
    model_config = config["model"]
    count = len(config["depths"]["depths_m"])
    if model_config["depth_weighting"] == "uniform":
        weights = np.ones(count)
    elif model_config["depth_weighting"] == "explicit":
        weights = np.asarray(model_config["depth_weights"], dtype=float)
    else:
        weights = 1.0 / np.asarray(stats["target_temperature"]["std"], dtype=float)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def _loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    squared_error = (prediction - target) ** 2
    weighted = squared_error * weights.to(prediction.device)[None, :, None, None]
    weighted = weighted * mask[:, None, :, :]
    return weighted.sum() / (mask.sum() * prediction.shape[1]).clamp_min(1.0)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    weights: torch.Tensor,
    device: torch.device,
) -> tuple[float, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    losses = []
    depth_errors = []
    for batch in loader:
        inputs, targets, mask = batch
        inputs, targets, mask = inputs.to(device), targets.to(device), mask.to(device)
        with torch.set_grad_enabled(training):
            predictions, _ = model(inputs)
            loss = _loss(predictions, targets, mask, weights)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        losses.append(float(loss.detach().cpu()))
        errors = ((predictions.detach() - targets) ** 2) * mask[:, None, :, :]
        depth_errors.append((errors.sum(dim=(0, 2, 3)) / mask.sum(dim=(1, 2)).clamp_min(1.0).sum()).cpu().numpy())
    if not losses:
        return float("nan"), np.full(weights.numel(), np.nan)
    return float(np.mean(losses)), np.sqrt(np.mean(depth_errors, axis=0))


def train(config_path: str | Path = "config.yaml", epochs: int | None = None) -> Path:
    config = load_config(config_path)
    seed = config["model"]["random_seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Training on device: {device} ({device_name})", flush=True)

    stats = load_normalization_stats(config_path)
    train_dataset = OceanEmbedDataset("train", config_path, return_mask=True)
    val_dataset = OceanEmbedDataset("val", config_path, return_mask=True)

    train_subset = Subset(train_dataset, train_dataset.valid_indices)
    val_subset = Subset(val_dataset, val_dataset.valid_indices)

    pin_mem = torch.cuda.is_available()
    train_loader = DataLoader(
        train_subset,
        batch_size=config["model"]["batch_size"],
        shuffle=True,
        pin_memory=pin_mem,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=config["model"]["batch_size"],
        shuffle=False,
        pin_memory=pin_mem,
    )

    if not len(train_subset) or not len(val_subset):
        raise RuntimeError("Training requires finite surface/target samples in both train and val periods")

    print(
        f"Reconstruction dataset: {len(train_subset)} train samples, {len(val_subset)} val samples",
        flush=True,
    )

    height, width = train_dataset.surface.sizes["lat"], train_dataset.surface.sizes["lon"]
    input_channels = len(train_dataset[0][0])
    output_channels = len(config["depths"]["depths_m"])

    model = OceanEmbedReconstructionModel(
        input_channels,
        output_channels,
        config["model"]["embedding_dim"],
        config["model"]["base_channels"],
    ).to(device)

    pretrained = Path(config["paths"]["model_checkpoint_dir"]) / "surface_autoencoder.pt"
    if config["model"]["use_pretrained_encoder"] and pretrained.exists():
        checkpoint = torch.load(pretrained, map_location="cpu", weights_only=True)
        model.encoder.load_state_dict(checkpoint["encoder_state_dict"])
        print(f"Successfully loaded pretrained encoder from {pretrained}", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["model"]["learning_rate"])
    weights = _depth_weights(config, stats).to(device)
    history: dict[str, list[Any]] = {"train_loss": [], "val_loss": [], "val_rmse": [], "epoch_duration_s": []}
    best_loss = float("inf")

    checkpoint_dir = Path(config["paths"]["model_checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = checkpoint_dir / "reconstruction_model_best.pt"
    latest_checkpoint_path = checkpoint_dir / "reconstruction_model_latest.pt"

    total_epochs = epochs or config["model"]["epochs"]
    print(f"Beginning reconstruction model training for {total_epochs} epochs...", flush=True)

    for epoch in range(total_epochs):
        t0 = time.perf_counter()
        train_loss, _ = _run_epoch(model, train_loader, optimizer, weights, device)
        val_loss, val_rmse = _run_epoch(model, val_loader, None, weights, device)
        duration = time.perf_counter() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_rmse"].append(val_rmse.tolist())
        history["epoch_duration_s"].append(duration)

        saved_best = ""
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_channels": input_channels,
                    "output_channels": output_channels,
                    "embedding_dim": config["model"]["embedding_dim"],
                    "base_channels": config["model"]["base_channels"],
                    "output_size": (height, width),
                    "history": history,
                    "epoch": epoch + 1,
                    "best_loss": best_loss,
                },
                best_checkpoint_path,
            )
            saved_best = " [BEST SAVED]"

        # Always save latest checkpoint after every epoch
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "input_channels": input_channels,
                "output_channels": output_channels,
                "embedding_dim": config["model"]["embedding_dim"],
                "base_channels": config["model"]["base_channels"],
                "output_size": (height, width),
                "history": history,
                "epoch": epoch + 1,
            },
            latest_checkpoint_path,
        )

        # Write history continuously
        (checkpoint_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

        mean_rmse = float(np.mean(val_rmse))
        print(
            f"epoch {epoch + 1}/{total_epochs} ({duration:.1f}s): "
            f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, "
            f"val_rmse_mean={mean_rmse:.3f}{saved_best}",
            flush=True,
        )

    print(f"Training completed. Best model saved to {best_checkpoint_path} (best val_loss={best_loss:.6f})", flush=True)
    return best_checkpoint_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    args = parser.parse_args()
    train(args.config, epochs=args.epochs)