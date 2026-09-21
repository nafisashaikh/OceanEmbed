"""Evaluate the trained reconstruction model against monthly climatology."""

from __future__ import annotations

from pathlib import Path
import json
import sys
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.climatology import evaluate_climatology
from src.config import load_config
from src.data.dataset import OceanEmbedDataset, load_normalization_stats
from src.models.reconstruction_model import OceanEmbedReconstructionModel


def _model_rmse(model: torch.nn.Module, dataset: OceanEmbedDataset, config: dict, device: torch.device) -> dict[float, float]:
    subset = Subset(dataset, dataset.valid_indices)
    if not len(subset):
        return {float(depth): float("nan") for depth in config["depths"]["depths_m"]}
    loader = DataLoader(subset, batch_size=config["model"]["batch_size"], shuffle=False)
    stats = load_normalization_stats()
    means = torch.tensor(stats["target_temperature"]["mean"], dtype=torch.float32, device=device)[None, :, None, None]
    stds = torch.tensor(stats["target_temperature"]["std"], dtype=torch.float32, device=device)[None, :, None, None]
    errors = []
    model.eval()
    with torch.no_grad():
        for inputs, targets, mask in loader:
            prediction, _ = model(inputs.to(device))
            physical_prediction = prediction * stds + means
            physical_target = targets.to(device) * stds + means
            squared_error = ((physical_prediction - physical_target) ** 2) * mask.to(device)[:, None, :, :]
            errors.append((squared_error.sum(dim=(0, 2, 3)) / mask.to(device).sum(dim=(1, 2)).clamp_min(1.0).sum()).cpu().numpy())
    return {float(depth): float(np.sqrt(np.mean(errors, axis=0)[index])) for index, depth in enumerate(config["depths"]["depths_m"])}


def main(config_path: str | Path = "config.yaml") -> int:
    config = load_config(config_path)
    checkpoint_dir = Path(config["paths"]["model_checkpoint_dir"])
    checkpoint_path = checkpoint_dir / "reconstruction_model_best.pt"
    history_path = checkpoint_dir / "training_history.json"
    if not checkpoint_path.exists() or not history_path.exists():
        print("FAIL: train the reconstruction model before evaluation")
        return 1
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    history = json.loads(history_path.read_text(encoding="utf-8"))
    figure, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set(title="Reconstruction loss", xlabel="epoch", ylabel="MSE")
    axes[0].legend()
    val_rmse = np.asarray(history["val_rmse"])
    for index, depth in enumerate(config["depths"]["depths_m"]):
        axes[1].plot(val_rmse[:, index], label=f"{depth:g} m")
    axes[1].set(title="Per-depth validation RMSE", xlabel="epoch", ylabel="RMSE (normalized)")
    axes[1].legend(fontsize="small", ncol=2)
    figure.tight_layout()
    output_dir = Path(config["paths"]["sanity_output"]).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / "training_curves_and_depth_rmse.png", dpi=150)
    plt.close(figure)

    train_dataset = OceanEmbedDataset("train", config_path)
    model = OceanEmbedReconstructionModel(checkpoint["input_channels"], checkpoint["output_channels"], checkpoint["embedding_dim"], checkpoint["base_channels"])
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    val_rmse_physical = _model_rmse(model, OceanEmbedDataset("val", config_path, return_mask=True), config, device)
    test_rmse_physical = _model_rmse(model, OceanEmbedDataset("test", config_path, return_mask=True), config, device)
    val_climatology = evaluate_climatology("val", config_path)
    test_climatology = evaluate_climatology("test", config_path)
    print("depth_m | model_val_rmse | climatology_val_rmse | model_test_rmse | climatology_test_rmse")
    for depth in config["depths"]["depths_m"]:
        depth = float(depth)
        print(f"{depth:7.1f} | {val_rmse_physical[depth]:15.5f} | {val_climatology[depth]:21.5f} | {test_rmse_physical[depth]:15.5f} | {test_climatology[depth]:23.5f}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    raise SystemExit(main(parser.parse_args().config))