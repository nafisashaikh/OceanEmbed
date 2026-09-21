"""Test the pretrained surface embedding and held-out reconstruction quality."""

from __future__ import annotations

from pathlib import Path
import sys
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import torch
from torch.utils.data import DataLoader
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data.dataset import INPUT_VARIABLES, load_normalization_stats
from src.models.autoencoder_pretrain import SurfaceOnlyDataset, _available_indices
from src.models.embedding_encoder import SurfaceAutoencoder


def main(config_path: str | Path = "config.yaml", use_tsne: bool = False) -> int:
    config = load_config(config_path)
    checkpoint_path = Path(config["paths"]["model_checkpoint_dir"]) / "surface_autoencoder.pt"
    if not checkpoint_path.exists():
        print(f"FAIL: pretrained checkpoint not found: {checkpoint_path}")
        return 1

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    surface = xr.open_zarr(config["paths"]["processed_data_dir"]).load()
    stats = load_normalization_stats(config_path)

    train_end = np.searchsorted(
        surface.time.values,
        np.datetime64(config["time"]["train_end"].replace("Z", "")),
        side="right",
    )
    val_end = np.searchsorted(
        surface.time.values,
        np.datetime64(config["time"]["val_end"].replace("Z", "")),
        side="right",
    )
    dates_total = len(surface.time)

    all_indices = _available_indices(surface, 0, dates_total, config["time"]["lag_days"])
    val_indices = _available_indices(surface, train_end, val_end, config["time"]["lag_days"])
    heldout_indices = _available_indices(surface, train_end, dates_total, config["time"]["lag_days"])

    if not all_indices:
        print("FAIL: no finite surface days are available for evaluation")
        surface.close()
        return 1

    model = SurfaceAutoencoder(
        checkpoint["input_channels"],
        checkpoint["embedding_dim"],
        tuple(checkpoint["output_size"]),
        checkpoint["base_channels"],
    )
    model.encoder.load_state_dict(checkpoint["encoder_state_dict"])
    model.decoder.load_state_dict(checkpoint["decoder_state_dict"])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 1. Evaluate on all available surface days
    dataset = SurfaceOnlyDataset(surface, all_indices, stats, config["time"]["lag_days"])
    loader = DataLoader(dataset, batch_size=config["model"]["batch_size"], shuffle=False)

    embeddings = []
    reconstruction_errors = []
    with torch.no_grad():
        for inputs in loader:
            inputs = inputs.to(device)
            reconstruction, embedding = model(inputs)
            embeddings.append(embedding.cpu().numpy())
            mse = torch.mean((reconstruction - inputs) ** 2, dim=(1, 2, 3))
            reconstruction_errors.extend(mse.cpu().numpy().tolist())

    embedding_array = np.concatenate(embeddings, axis=0)
    months = np.array([pd.Timestamp(surface.time.values[idx]).month for idx in all_indices])
    
    heldout_set = set(heldout_indices)
    is_heldout = np.array([idx in heldout_set for idx in all_indices])

    # 2. 2D Dimensionality Reduction (PCA or t-SNE)
    method_label = "t-SNE" if use_tsne else "PCA"
    if use_tsne:
        from sklearn.manifold import TSNE
        perplexity = min(30, max(5, len(embedding_array) // 3))
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=config["model"]["random_seed"])
    else:
        reducer = PCA(n_components=2, random_state=config["model"]["random_seed"])

    coordinates = reducer.fit_transform(embedding_array)

    # 3. Plot embeddings colored by month
    figure, axis = plt.subplots(figsize=(8, 6))
    
    train_mask = ~is_heldout
    cmap_name = "tab10" if len(set(months)) > 1 else "viridis"
    
    # In-sample (train) points - no outline
    scatter_train = axis.scatter(
        coordinates[train_mask, 0],
        coordinates[train_mask, 1],
        c=months[train_mask],
        cmap=cmap_name,
        vmin=1, vmax=12,
        s=40,
        alpha=0.4,
        edgecolor="none",
        label="Train (in-sample)",
    )
    
    # Held-out points - distinct outline
    if np.any(is_heldout):
        axis.scatter(
            coordinates[is_heldout, 0],
            coordinates[is_heldout, 1],
            c=months[is_heldout],
            cmap=cmap_name,
            vmin=1, vmax=12,
            s=60,
            alpha=0.9,
            edgecolor="black",
            linewidth=1.5,
            label="Held-out (Dec '23, Jan '24)",
        )
        
    colorbar = figure.colorbar(scatter_train, ax=axis, label="Calendar Month")
    unique_months = sorted(set(months))
    colorbar.set_ticks(unique_months)
    colorbar.set_ticklabels([f"{m}" for m in unique_months])

    axis.legend(loc="best", fontsize=9)
    axis.set(
        title=f"OceanEmbed Latent Space ({method_label} of All {dates_total} Days)\n"
              f"Note: Training points are in-sample. Illustrates representation structure.",
        xlabel=f"{method_label} Component 1",
        ylabel=f"{method_label} Component 2",
    )
    axis.grid(True, linestyle="--", alpha=0.5)
    figure.tight_layout()

    sanity_output_dir = Path(config["paths"]["sanity_output"]).parent
    sanity_output_dir.mkdir(parents=True, exist_ok=True)
    output_plot = sanity_output_dir / f"embedding_{method_label.lower()}_by_month.png"
    figure.savefig(output_plot, dpi=150)
    plt.close(figure)
    
    dates_str = [pd.Timestamp(surface.time.values[idx]).strftime("%Y-%m-%d") for idx in all_indices]
    csv_df = pd.DataFrame({
        "date": dates_str,
        "month": months,
        "PC1": coordinates[:, 0],
        "PC2": coordinates[:, 1],
        "is_held_out": is_heldout
    })
    csv_path = sanity_output_dir / f"embedding_{method_label.lower()}_coords.csv"
    csv_df.to_csv(csv_path, index=False)

    val_count = len(val_indices)
    
    val_set = set(val_indices)
    val_errors = [err for err, idx in zip(reconstruction_errors, all_indices) if idx in val_set]
    heldout_errors = [err for err, h in zip(reconstruction_errors, is_heldout) if h]
    
    val_mse = np.mean(val_errors) if val_count > 0 else float("nan")
    heldout_mse = float(np.mean(heldout_errors)) if heldout_errors else float("nan")

    print(f"PASS: {method_label} embedding plot saved to {output_plot}")
    print(f"val-period surface days: {val_count} (reconstruction MSE: {val_mse:.6f})")
    print(f"total held-out surface days: {len(heldout_indices)}")
    print(f"held-out reconstruction MSE: {heldout_mse:.6f}")

    surface.close()
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--tsne", action="store_true", help="Use t-SNE instead of PCA")
    args = parser.parse_args()
    raise SystemExit(main(args.config, use_tsne=args.tsne))