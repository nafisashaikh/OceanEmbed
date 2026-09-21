"""Pretrain the surface encoder with an autoencoder reconstruction objective."""

from __future__ import annotations

from pathlib import Path
import argparse
import random
import sys
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data.dataset import INPUT_VARIABLES, load_normalization_stats
from src.models.embedding_encoder import SurfaceAutoencoder


class SurfaceOnlyDataset(Dataset):
    """Surface-only days with fast in-memory array slicing."""

    def __init__(self, surface: xr.Dataset, indices: list[int], stats: dict[str, Any], lag_days: int):
        self.surface = surface
        self.indices = indices
        self.stats = stats
        self.lag_days = lag_days

        # Pre-cache numpy arrays in memory for high-throughput batching
        self.surface_arrays = {
            name: surface[name].values.astype(np.float32) for name in INPUT_VARIABLES
        }
        self.means = {
            name: np.float32(stats["inputs"][name]["mean"]) for name in INPUT_VARIABLES
        }
        self.stds = {
            name: np.float32(stats["inputs"][name]["std"]) for name in INPUT_VARIABLES
        }

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> torch.Tensor:
        current = self.indices[index]
        channels = []
        for lag in range(self.lag_days, -1, -1):
            day_idx = current - lag
            for name in INPUT_VARIABLES:
                val = (self.surface_arrays[name][day_idx] - self.means[name]) / self.stds[name]
                channels.append(np.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0))
        return torch.from_numpy(np.stack(channels))


def _available_indices(surface: xr.Dataset, start: int, end: int, lag_days: int) -> list[int]:
    """Find all valid days with finite surface pixels in [start, end)."""
    time_slice = surface.isel(time=slice(start, end))
    has_finite = np.zeros(len(time_slice.time), dtype=bool)
    for name in INPUT_VARIABLES:
        has_finite |= np.isfinite(time_slice[name].values).any(axis=(1, 2))

    indices = []
    for local_idx, is_finite in enumerate(has_finite):
        global_idx = start + local_idx
        if global_idx - lag_days >= 0 and is_finite:
            indices.append(global_idx)
    return indices


def train_autoencoder(config_path: str | Path = "config.yaml", epochs: int | None = None) -> Path:
    config = load_config(config_path)
    seed = config["model"]["random_seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Using device: {device} ({device_name})", flush=True)

    surface = xr.open_zarr(config["paths"]["processed_data_dir"]).load()
    stats = load_normalization_stats(config_path)
    dates = len(surface.time)
    train_end = np.searchsorted(
        surface.time.values,
        np.datetime64(config["time"]["train_end"].replace("Z", "")),
        side="right",
    )
    train_indices = _available_indices(surface, 0, train_end, config["time"]["lag_days"])
    heldout_indices = _available_indices(surface, train_end, dates, config["time"]["lag_days"])
    if not train_indices:
        raise RuntimeError("No finite surface-only days are available for autoencoder pretraining")

    print(
        f"Surface pretraining dataset: {len(train_indices)} training days, "
        f"{len(heldout_indices)} held-out days",
        flush=True,
    )

    height, width = surface.sizes["lat"], surface.sizes["lon"]
    input_channels = len(INPUT_VARIABLES) * (config["time"]["lag_days"] + 1)
    model = SurfaceAutoencoder(
        input_channels,
        config["model"]["embedding_dim"],
        (height, width),
        config["model"]["base_channels"],
    ).to(device)

    train_dataset = SurfaceOnlyDataset(surface, train_indices, stats, config["time"]["lag_days"])
    loader = DataLoader(
        train_dataset,
        batch_size=config["model"]["batch_size"],
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config["model"]["learning_rate"])
    criterion = nn.MSELoss()

    total_epochs = epochs or config["model"].get("pretrain_epochs") or config["model"]["epochs"]
    print(f"Starting autoencoder pretraining for {total_epochs} epochs...", flush=True)

    model.train()
    for epoch in range(total_epochs):
        losses = []
        for inputs in loader:
            inputs = inputs.to(device)
            reconstruction, _ = model(inputs)
            loss = criterion(reconstruction, inputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        avg_loss = np.mean(losses)
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch + 1 == total_epochs:
            print(f"epoch {epoch + 1}/{total_epochs}: reconstruction_mse={avg_loss:.6f}", flush=True)

    checkpoint_dir = Path(config["paths"]["model_checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / "surface_autoencoder.pt"
    torch.save(
        {
            "encoder_state_dict": model.encoder.state_dict(),
            "decoder_state_dict": model.decoder.state_dict(),
            "input_channels": input_channels,
            "embedding_dim": config["model"]["embedding_dim"],
            "base_channels": config["model"]["base_channels"],
            "output_size": (height, width),
            "train_days": len(train_indices),
            "heldout_days": len(heldout_indices),
        },
        checkpoint,
    )
    print(f"Saved pretrained autoencoder checkpoint to {checkpoint}", flush=True)
    surface.close()
    return checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    train_autoencoder(args.config, epochs=args.epochs)