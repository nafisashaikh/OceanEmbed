"""Validate target alignment and calculate the GLORYS ceiling against ARGO."""

from __future__ import annotations

from pathlib import Path
import sys
import argparse

import numpy as np
import pandas as pd
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data.prepare_argo_validation import prepare_argo_validation
from src.data.prepare_target import prepare_target


def _metric_line(depth: float, observed: np.ndarray, predicted: np.ndarray) -> None:
    valid = np.isfinite(observed) & np.isfinite(predicted)
    observed = observed[valid]
    predicted = predicted[valid]
    if not len(observed):
        print(f"depth={depth:g} m: no matched observations")
        return
    rmse = float(np.sqrt(np.mean((predicted - observed) ** 2)))
    correlation = float(np.corrcoef(observed, predicted)[0, 1]) if len(observed) > 1 else float("nan")
    print(f"depth={depth:g} m: n={len(observed)}, RMSE={rmse:.4f} C, correlation={correlation:.4f}")


def main(config_path: str | Path = "config.yaml") -> int:
    config = load_config(config_path)
    surface_path = Path(config["paths"]["processed_data_dir"])
    target_path = surface_path.parent / "target_temperature.zarr"
    argo_path = Path(config["argo"]["output_validation_table"])
    if not target_path.exists():
        prepare_target(config_path)
    if not argo_path.exists():
        prepare_argo_validation(config_path)

    surface = xr.open_zarr(surface_path)
    target = xr.open_zarr(target_path)
    aligned = (
        np.array_equal(surface.time.values, target.time.values)
        and np.allclose(surface.lat.values, target.lat.values)
        and np.allclose(surface.lon.values, target.lon.values)
    )
    print(f"{'PASS' if aligned else 'FAIL'}: target and surface share identical time/lat/lon")

    argo = pd.read_csv(argo_path, parse_dates=["time"])
    validation = argo[argo["split"] == "validation"].copy()
    has_platforms = argo["platform_number"].notna().all() and (argo["platform_number"].astype(str).str.len() > 0).all()
    print(f"{'PASS' if has_platforms else 'FAIL'}: validation ARGO records have real platform numbers")
    print(f"held-out validation floats: {validation['platform_number'].nunique()}")

    for depth in config["depths"]["depths_m"]:
        subset = validation.iloc[(validation["depth"] - depth).abs().argsort()[: len(validation)]]
        subset = subset[(subset["depth"] - depth).abs() <= max(5.0, depth * 0.1)]
        if subset.empty:
            _metric_line(depth, np.array([]), np.array([]))
            continue
        predictions = target["target_temperature"].sel(
            depth=depth,
            time=xr.DataArray(subset["time"].dt.tz_localize(None).to_numpy(), dims="observation"),
            lat=xr.DataArray(subset["lat"].to_numpy(), dims="observation"),
            lon=xr.DataArray(subset["lon"].to_numpy(), dims="observation"),
            method="nearest",
        ).values
        _metric_line(depth, subset["temperature"].to_numpy(), predictions)

    surface.close()
    target.close()
    return 0 if aligned and has_platforms else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    raise SystemExit(main(parser.parse_args().config))