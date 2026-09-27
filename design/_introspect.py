import json
from pathlib import Path
import numpy as np
import xarray as xr
import torch

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "processed_data"

def show(name, ds):
    print(f"\n=== {name} ===")
    print("dims:", dict(ds.sizes))
    print("data_vars:", list(ds.data_vars))
    print("coords:", list(ds.coords))
    if "time" in ds.coords:
        t = ds.time.values
        print("time n=", len(t), "first=", str(t[0])[:10], "last=", str(t[-1])[:10])
    for c in ("lat", "lon", "depth"):
        if c in ds.coords:
            v = ds[c].values
            print(f"{c}: n={len(v)} min={float(v.min()):.3f} max={float(v.max()):.3f}")

surf = xr.open_zarr(PROC / "oceanembed.zarr")
show("oceanembed.zarr (surface inputs)", surf)

pred = xr.open_zarr(PROC / "predicted_field.zarr")
show("predicted_field.zarr", pred)
print("predicted times:", [str(t)[:10] for t in pred.time.values])

ck = torch.load(ROOT / "checkpoints/oceanembed/reconstruction_model_best.pt", map_location="cpu", weights_only=False)
print("\n=== checkpoint reconstruction_model_best.pt ===")
for k in ("input_channels", "output_channels", "embedding_dim", "base_channels", "output_size", "epoch", "best_loss"):
    print(k, "=", ck.get(k))
print("state_dict n keys:", len(ck["model_state_dict"]))
