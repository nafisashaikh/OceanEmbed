"""Train-period monthly climatology baseline for target temperatures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from src.config import load_config


def monthly_climatology(config_path: str | Path = "config.yaml") -> xr.DataArray:
    config = load_config(config_path)
    target_path = Path(config["paths"]["processed_data_dir"]).parent / "target_temperature.zarr"
    target = xr.open_zarr(target_path)["target_temperature"]
    train_end = pd.Timestamp(config["time"]["train_end"]).tz_localize(None)
    train = target.sel(time=target.time <= np.datetime64(train_end))
    if train.sizes.get("time", 0) == 0:
        raise ValueError("No target days exist in the configured training period")
    return train.groupby("time.month").mean("time", skipna=True)


def evaluate_climatology(split: str, config_path: str | Path = "config.yaml") -> dict[float, float]:
    if split not in {"val", "test"}:
        raise ValueError("split must be val or test")
    config = load_config(config_path)
    surface = xr.open_zarr(config["paths"]["processed_data_dir"])
    target = xr.open_zarr(Path(config["paths"]["processed_data_dir"]).parent / "target_temperature.zarr")["target_temperature"]
    times = pd.DatetimeIndex(surface.time.values)
    train_end = pd.Timestamp(config["time"]["train_end"]).tz_localize(None)
    val_end = pd.Timestamp(config["time"]["val_end"]).tz_localize(None)
    mask = (times > train_end) & (times <= val_end) if split == "val" else times > val_end
    selected = target.isel(time=mask)
    climatology = monthly_climatology(config_path)
    values = {}
    for depth in target.depth.values:
        errors = []
        for index, timestamp in enumerate(selected.time.values):
            prediction = climatology.sel(month=pd.Timestamp(timestamp).month, depth=depth).values
            observed = selected.isel(time=index).sel(depth=depth).values
            finite = np.isfinite(prediction) & np.isfinite(observed)
            if finite.any():
                errors.extend((prediction[finite] - observed[finite]).tolist())
        values[float(depth)] = float(np.sqrt(np.mean(np.square(errors)))) if errors else float("nan")
    surface.close()
    target.close()
    return values