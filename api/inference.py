"""Live OceanEmbed U-Net reconstruction.

Loads the trained reconstruction model once and runs a genuine forward pass on
the satellite surface inputs for a requested date, reproducing the exact input
assembly / normalization used in training (``OceanEmbedDataset.__getitem__``).

Nothing here is faked: the model, the normalization statistics and the surface
inputs are the same artifacts the offline pipeline produced.
"""
from __future__ import annotations

import json
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "processed_data"
CKPT = ROOT / "checkpoints" / "oceanembed" / "reconstruction_model_best.pt"
SURFACE_ZARR = PROC / "oceanembed.zarr"
PRED_ZARR = PROC / "predicted_field.zarr"

# 8 satellite/reanalysis surface inputs, in the exact order used at train time.
INPUT_VARIABLES = ("sst", "sss", "ssh", "temperature",
                   "current_u", "current_v", "wind_u", "wind_v")
LAG_DAYS = 1  # input = [t-1, t] stacked -> 8 * 2 = 16 channels

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@lru_cache(maxsize=1)
def stats() -> dict:
    return json.loads((PROC / "norm_stats.json").read_text())


@lru_cache(maxsize=1)
def _model():
    from src.models.reconstruction_model import OceanEmbedReconstructionModel
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = OceanEmbedReconstructionModel(
        input_channels=ck["input_channels"],
        output_channels=ck["output_channels"],
        embedding_dim=ck["embedding_dim"],
        base_channels=ck["base_channels"],
    )
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model, ck


@lru_cache(maxsize=1)
def _surface():
    return xr.open_zarr(SURFACE_ZARR)


@lru_cache(maxsize=1)
def _times() -> np.ndarray:
    return np.array([str(t)[:10] for t in _surface().time.values])


def model_meta() -> dict:
    _, ck = _model()
    n_params = sum(p.numel() for p in _model()[0].parameters())
    return {
        "input_channels": int(ck["input_channels"]),
        "output_channels": int(ck["output_channels"]),
        "embedding_dim": int(ck["embedding_dim"]),
        "base_channels": int(ck["base_channels"]),
        "output_size": list(ck.get("output_size", [101, 241])),
        "epoch": int(ck.get("epoch", -1)),
        "parameters": int(n_params),
        "input_variables": list(INPUT_VARIABLES),
        "lag_days": LAG_DAYS,
    }


def available_dates() -> list[str]:
    """Test-period dates (after val_end 2023-12-31) we can reconstruct live."""
    return [t for t in _times().tolist() if t >= "2024-01-01"]


def _build_input(date: str):
    """Assemble the 16-channel normalized input tensor for ``date``.

    Mirrors OceanEmbedDataset: for each lag in [t-1, t], every input variable is
    standardized with its train-set mean/std, land/NaN -> 0, then stacked.
    """
    surf = _surface()
    st = stats()["inputs"]
    times = _times()
    hits = np.where(times == date)[0]
    if hits.size == 0:
        raise KeyError(date)
    idx = int(hits[0])
    if idx - LAG_DAYS < 0:
        raise ValueError(f"no lag-{LAG_DAYS} history available for {date}")

    channels = []
    for lag in range(LAG_DAYS, -1, -1):          # t-1 first, then t
        di = idx - lag
        for name in INPUT_VARIABLES:
            arr = surf[name].isel(time=di).values.astype(np.float32)
            arr = (arr - st[name]["mean"]) / st[name]["std"]
            channels.append(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0))
    x = np.stack(channels)                        # (16, H, W)
    mask = surf["ocean_mask"].isel(time=idx).values.astype(bool)
    return torch.from_numpy(x[None]), mask


def reconstruct(date: str):
    """Run the U-Net live for ``date``; return per-depth denormalized field.

    Returns (field[15,H,W] with NaN over land, depths[15], latency_ms, embedding_norm).
    """
    model, _ = _model()
    tgt = stats()["target_temperature"]
    x, mask = _build_input(date)

    t0 = time.perf_counter()
    with torch.no_grad():
        out, emb = model(x)                       # (1,15,H,W), (1,emb)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    out = out[0].numpy()
    mean = np.asarray(tgt["mean"], dtype=np.float32)[:, None, None]
    std = np.asarray(tgt["std"], dtype=np.float32)[:, None, None]
    field = out * std + mean                      # denormalize per depth
    field = np.where(mask[None], field, np.nan).astype(np.float32)
    emb_norm = float(np.linalg.norm(emb[0].numpy()))
    return field, tgt["depths"], latency_ms, emb_norm


def grid_axes():
    surf = _surface()
    return surf["lat"].values.astype(float), surf["lon"].values.astype(float)
