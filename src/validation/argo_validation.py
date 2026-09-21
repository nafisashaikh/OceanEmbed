
"""Validate test-period OceanEmbed predictions against real ARGO platform IDs."""

from __future__ import annotations

from pathlib import Path
import json
import os
import shutil
import uuid
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import ttest_rel
from torch.utils.data import DataLoader, Subset
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.climatology import monthly_climatology
from src.config import load_config
from src.data.dataset import INPUT_VARIABLES, OceanEmbedDataset, load_normalization_stats
from src.models.reconstruction_model import OceanEmbedReconstructionModel


def _replace_zarr_store(output: Path) -> None:
    """Remove prior generated stores, including read-only Windows Zarr chunks."""
    if not output.exists():
        return
    try:
        os.chmod(output, 0o777)
    except OSError:
        pass
    for path in output.rglob("*"):
        try:
            os.chmod(path, 0o666 if path.is_file() else 0o777)
        except OSError:
            pass
    try:
        shutil.rmtree(output)
    except PermissionError as error:
        raise PermissionError(
            f"Cannot replace generated Zarr store {output}. Close programs holding files in it and retry."
        ) from error


def _metric(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    valid = np.isfinite(observed) & np.isfinite(predicted)
    observed = observed[valid]
    predicted = predicted[valid]
    if not len(observed):
        return {"RMSE": float("nan"), "MAE": float("nan"), "bias": float("nan"), "correlation": float("nan"), "n_matches": 0}
    residual = predicted - observed
    correlation = float(np.corrcoef(observed, predicted)[0, 1]) if len(observed) > 1 and np.std(observed) > 0 and np.std(predicted) > 0 else float("nan")
    return {
        "RMSE": float(np.sqrt(np.mean(residual**2))),
        "MAE": float(np.mean(np.abs(residual))),
        "bias": float(np.mean(residual)),
        "correlation": correlation,
        "n_matches": int(len(observed)),
    }


def _empty_metric(depths: list[float]) -> dict[str, dict[str, float | int]]:
    return {str(depth): _metric(np.array([]), np.array([])) for depth in depths}


def _load_model(config: dict[str, Any]) -> tuple[torch.nn.Module, torch.device]:
    checkpoint_path = Path(config["paths"]["model_checkpoint_dir"]) / "reconstruction_model_best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing trained checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = OceanEmbedReconstructionModel(
        checkpoint["input_channels"], checkpoint["output_channels"], checkpoint["embedding_dim"], checkpoint["base_channels"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device).eval(), device


def generate_predicted_field(config_path: str | Path = "config.yaml") -> Path:
    """Generate physical-Celsius model predictions for every test day."""
    config = load_config(config_path)
    surface = xr.open_zarr(config["paths"]["processed_data_dir"]).load()
    stats = load_normalization_stats(config_path)
    model, device = _load_model(config)

    # Resolve test date range
    val_end = pd.Timestamp(config["time"]["val_end"])
    if val_end.tzinfo is not None:
        val_end = val_end.tz_convert(None)
    surface_times = pd.DatetimeIndex(surface.time.values)
    test_indices = [idx for idx in range(len(surface_times)) if surface_times[idx] > val_end]

    lag_days = config["time"]["lag_days"]
    valid_test_indices = [idx for idx in test_indices if idx - lag_days >= 0]
    if not valid_test_indices:
        raise RuntimeError("No valid test samples are available for prediction")

    means = {name: np.float32(stats["inputs"][name]["mean"]) for name in INPUT_VARIABLES}
    stds = {name: np.float32(stats["inputs"][name]["std"]) for name in INPUT_VARIABLES}
    surface_arrays = {name: surface[name].values.astype(np.float32) for name in INPUT_VARIABLES}
    mask_array = surface["ocean_mask"].values.astype(np.float32) if "ocean_mask" in surface else None

    target_means = torch.tensor(stats["target_temperature"]["mean"], dtype=torch.float32, device=device)[None, :, None, None]
    target_stds = torch.tensor(stats["target_temperature"]["std"], dtype=torch.float32, device=device)[None, :, None, None]

    batch_size = config["model"]["batch_size"]
    predictions = []

    with torch.no_grad():
        for b_start in range(0, len(valid_test_indices), batch_size):
            batch_idxs = valid_test_indices[b_start : b_start + batch_size]
            batch_inputs = []
            batch_masks = []
            for current in batch_idxs:
                channels = []
                for lag in range(lag_days, -1, -1):
                    day_idx = current - lag
                    for name in INPUT_VARIABLES:
                        val = (surface_arrays[name][day_idx] - means[name]) / stds[name]
                        channels.append(np.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0))
                batch_inputs.append(np.stack(channels))
                if mask_array is not None:
                    batch_masks.append(mask_array[current])

            inp_tensor = torch.from_numpy(np.stack(batch_inputs)).to(device)
            output, _ = model(inp_tensor)
            output = output * target_stds + target_means

            if mask_array is not None:
                mask_t = torch.from_numpy(np.stack(batch_masks)).to(device)[:, None, :, :].bool()
                output = torch.where(mask_t, output, torch.full_like(output, float("nan")))

            predictions.append(output.cpu().numpy())

    times = surface_times[valid_test_indices].to_numpy()
    output = Path(config["paths"]["processed_data_dir"]).parent / "predicted_field.zarr"
    temporary_output = output.parent / f"{output.name}.new_{uuid.uuid4().hex}"
    xr.Dataset(
        {"temperature": (("time", "depth", "lat", "lon"), np.concatenate(predictions))},
        coords={
            "time": times,
            "depth": config["depths"]["depths_m"],
            "lat": surface.lat.values,
            "lon": surface.lon.values,
        },
    ).to_zarr(temporary_output, mode="w")
    surface.close()
    try:
        _replace_zarr_store(output)
        temporary_output.replace(output)
        return output
    except PermissionError:
        return temporary_output


def _region_mask(table: pd.DataFrame, bounds: dict[str, float]) -> np.ndarray:
    return (
        table["lon"].between(bounds["west"], bounds["east"])
        & table["lat"].between(bounds["south"], bounds["north"])
    ).to_numpy()


def _match_record(config: dict[str, Any], predicted: xr.DataArray, glorys: xr.DataArray, climate: xr.DataArray, row: Any) -> dict[str, Any] | None:
    """Match one real ARGO record to model, GLORYS, and climatology fields."""
    depths = np.asarray(config["depths"]["depths_m"], dtype=float)
    depth_index = int(np.argmin(np.abs(depths - row.depth)))
    depth = float(depths[depth_index])
    if abs(float(row.depth) - depth) > config["validation"]["depth_tolerance_m"]:
        return None
    pred_times = pd.DatetimeIndex(predicted.time.values)
    row_time = pd.Timestamp(row.time).tz_localize(None) if pd.Timestamp(row.time).tzinfo is not None else pd.Timestamp(row.time)
    time_index = int(np.argmin(np.abs(pred_times - row_time)))
    if abs(pred_times[time_index] - row_time) > pd.Timedelta(days=config["validation"]["time_tolerance_days"]):
        return None
    tolerance = config["validation"]["spatial_tolerance_deg"]
    lat_index = int(np.argmin(np.abs(predicted.lat.values - row.lat)))
    lon_index = int(np.argmin(np.abs(predicted.lon.values - row.lon)))
    if abs(float(predicted.lat.values[lat_index]) - row.lat) > tolerance or abs(float(predicted.lon.values[lon_index]) - row.lon) > tolerance:
        return None
    prediction = float(predicted.isel(time=time_index, depth=depth_index, lat=lat_index, lon=lon_index).values)
    ceiling = float(glorys.sel(time=predicted.time.values[time_index], depth=depth, lat=row.lat, lon=row.lon, method="nearest").values)
    baseline = float(climate.sel(month=pd.Timestamp(row.time).month, depth=depth, lat=row.lat, lon=row.lon, method="nearest").values)
    if not np.isfinite(prediction) or not np.isfinite(ceiling) or not np.isfinite(baseline) or not np.isfinite(row.temperature):
        return None
    return {"platform_number": str(row.platform_number), "time": row.time, "lat": row.lat, "lon": row.lon, "depth": depth, "observed": row.temperature, "model": prediction, "glorys": ceiling, "climatology": baseline}


def _match_table(config: dict[str, Any], predicted: xr.DataArray, glorys: xr.DataArray, climate: xr.DataArray) -> pd.DataFrame:
    argo_path = config["argo"]["output_validation_table"]
    table = pd.read_csv(argo_path, parse_dates=["time"])
    test_start = pd.Timestamp(config["time"]["val_end"], tz="UTC")
    test_end = pd.Timestamp(config["dates"]["end"], tz="UTC")
    table = table[(table["split"] == "validation") & (table["time"] > test_start) & (table["time"] <= test_end)].copy()
    if table.empty:
        return pd.DataFrame()
    records = []
    for row in table.itertuples(index=False):
        record = _match_record(config, predicted, glorys, climate, row)
        if record is not None:
            records.append(record)
    return pd.DataFrame(records)


def _metrics_by(table: pd.DataFrame, depths: list[float], group: str | None = None, prediction_column: str = "model") -> dict[str, dict[str, float | int]]:
    result = _empty_metric(depths)
    for depth in depths:
        subset = table[table["depth"] == depth] if group is None else table[table["group"] == group]
        if group is not None:
            subset = subset[subset["depth"] == depth]
        result[str(depth)] = _metric(subset["observed"].to_numpy(), subset[prediction_column].to_numpy()) if not subset.empty else result[str(depth)]
    return result


def validate(config_path: str | Path = "config.yaml") -> dict[str, Any]:
    config = load_config(config_path)
    predicted_path = generate_predicted_field(config_path)
    predicted = xr.open_zarr(predicted_path)["temperature"]
    target = xr.open_zarr(Path(config["paths"]["processed_data_dir"]).parent / "target_temperature.zarr")["target_temperature"]
    climate = monthly_climatology(config_path)
    matched = _match_table(config, predicted, target, climate)
    depths = [float(depth) for depth in config["depths"]["depths_m"]]
    if matched.empty:
        raise RuntimeError("No held-out ARGO profiles matched the test prediction field")
    matched["season"] = matched["time"].dt.month.map({12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM", 6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"})
    for name, bounds in config["validation"]["regions"].items():
        matched[name] = _region_mask(matched, bounds)
    summary = {
        "overall": _metrics_by(matched, depths),
        "climatology": _metrics_by(matched, depths, prediction_column="climatology"),
        "glorys_ceiling": _metrics_by(matched, depths, prediction_column="glorys"),
        "season": {},
        "region": {},
    }
    for season in ("DJF", "MAM", "JJA", "SON"):
        summary["season"][season] = _metrics_by(matched[matched["season"] == season], depths)
    for region in config["validation"]["regions"]:
        summary["region"][region] = _metrics_by(matched[matched[region]], depths)
    model_absolute_error = np.abs(matched["model"].to_numpy() - matched["observed"].to_numpy())
    climate_absolute_error = np.abs(matched["climatology"].to_numpy() - matched["observed"].to_numpy())
    test = ttest_rel(model_absolute_error, climate_absolute_error, nan_policy="omit")
    summary["paired_ttest"] = {
        "t_statistic": float(test.statistic),
        "p_value": float(test.pvalue),
        "n_matches": int(len(matched)),
        "mean_absolute_error_difference_model_minus_climatology": float(np.mean(model_absolute_error - climate_absolute_error)),
    }
    summary["paired_ttest_by_depth"] = {}
    for depth in depths:
        subset = matched[matched["depth"] == depth]
        model_error = np.abs(subset["model"].to_numpy() - subset["observed"].to_numpy())
        climate_error = np.abs(subset["climatology"].to_numpy() - subset["observed"].to_numpy())
        if len(model_error) >= 2:
            depth_test = ttest_rel(model_error, climate_error, nan_policy="omit")
            p_value = float(depth_test.pvalue)
            difference = float(np.mean(model_error - climate_error))
        else:
            p_value = float("nan")
            difference = float("nan")
        summary["paired_ttest_by_depth"][str(depth)] = {
            "p_value": p_value,
            "mean_absolute_error_difference_model_minus_climatology": difference,
            "n_matches": int(len(model_error)),
            "model_significantly_better": bool(np.isfinite(p_value) and p_value < 0.05 and difference < 0),
        }
    summary["platform_audit"] = {
        "all_matches_have_platform_number": bool(matched["platform_number"].notna().all() and (matched["platform_number"].astype(str).str.strip() != "").all()),
        "distinct_platforms": int(matched["platform_number"].nunique()),
        "matched_rows": int(len(matched)),
    }
    summary["matched_records"] = matched.to_dict(orient="records")
    output = Path(config["paths"]["processed_data_dir"]).parent / "argo_validation_summary.json"
    output.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    predicted.close()
    target.close()
    return summary