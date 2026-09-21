"""Regrid, daily-average, mask, and combine OceanEmbed source datasets."""

from __future__ import annotations

from pathlib import Path
import argparse
import gc
import os
import re
import shutil
import sys
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config


_SPATIAL_DIMS = {"lat", "latitude", "lon", "longitude"}


def _files_for_input(config: dict[str, Any], input_config: dict[str, Any]) -> list[Path]:
    raw_root = Path(config["paths"]["raw_data_dir"])
    files = sorted(raw_root.glob(input_config["glob"]))
    if not files:
        raise FileNotFoundError(
            f"No raw files matched {raw_root / input_config['glob']}"
        )
    return files


def _coordinate_name(data: xr.DataArray, candidates: tuple[str, ...], axis: str) -> str:
    for name in candidates:
        if name in data.coords:
            return name
    for name in data.dims:
        if name in candidates:
            return name
    raise ValueError(f"Could not find {axis} coordinate in {data.name}")


def _standardize_dataarray(
    dataset: xr.Dataset,
    input_config: dict[str, Any],
) -> xr.DataArray:
    variable = input_config["variable"]
    if variable not in dataset:
        raise KeyError(f"Variable {variable!r} is not present in dataset")
    data = dataset[variable]
    lat_name = _coordinate_name(data, ("lat", "latitude"), "latitude")
    lon_name = _coordinate_name(data, ("lon", "longitude"), "longitude")
    lat_dim = data[lat_name].dims[0] if lat_name in data.coords else lat_name
    lon_dim = data[lon_name].dims[0] if lon_name in data.coords else lon_name
    lat_values = np.asarray(data[lat_name].values)
    lon_values = np.asarray(data[lon_name].values)

    if lat_dim != "lat":
        if "lat" in data.coords:
            data = data.swap_dims({lat_dim: "lat"})
        else:
            data = data.rename({lat_dim: "lat"})
            if "lat" in data.coords and "lat" not in data.indexes:
                data = data.set_index(lat="lat")
    if lon_dim != "lon":
        if "lon" in data.coords:
            data = data.swap_dims({lon_dim: "lon"})
        else:
            data = data.rename({lon_dim: "lon"})
            if "lon" in data.coords and "lon" not in data.indexes:
                data = data.set_index(lon="lon")
    data = data.assign_coords(lat=("lat", lat_values), lon=("lon", lon_values))

    extra_dims = [dim for dim in data.dims if dim not in {"time", "lat", "lon"}]
    for dim in extra_dims:
        if input_config.get("depth") == "minimum":
            data = data.sel({dim: data[dim].min()}, method="nearest")
        elif data.sizes[dim] == 1:
            data = data.isel({dim: 0}, drop=True)
        else:
            raise ValueError(
                f"{variable} has unsupported non-singleton dimension {dim}; configure a depth rule"
            )

    if "time" not in data.dims:
        raise ValueError(f"{variable} must contain a time dimension")
    data = data.assign_coords(
        time=("time", pd.to_datetime([str(value) for value in data.time.values]))
    )
    for coordinate in list(data.coords):
        if coordinate not in {"time", "lat", "lon"}:
            data = data.drop_vars(coordinate)
    data = data.transpose("time", "lat", "lon")
    data = data.sortby("lat")
    data = data.assign_coords(lon=((data.lon + 180) % 360) - 180).sortby("lon")
    if input_config["units"] == "kelvin":
        data = data - 273.15
        data.attrs = {**data.attrs, "units": "degC"}
    elif input_config["units"] == "psu":
        data.attrs = {**data.attrs, "units": "psu"}
    elif input_config["units"] == "m":
        data.attrs = {**data.attrs, "units": "m"}
    elif input_config["units"] == "m_per_s":
        data.attrs = {**data.attrs, "units": "m s-1"}
    return data


def _month_key_from_path(path: Path) -> str:
    m = re.search(r"(\d{4}-\d{2})", path.stem)
    if m:
        return m.group(1)
    m_daily = re.search(r"(\d{4})(\d{2})(\d{2})", path.stem)
    if m_daily:
        return f"{m_daily.group(1)}-{m_daily.group(2)}"
    raise ValueError(f"Cannot extract YYYY-MM month from file path {path.name}")


def _group_files_by_month(files: list[Path]) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {}
    for path in files:
        month = _month_key_from_path(path)
        grouped.setdefault(month, []).append(path)
    return grouped


def _target_coordinates(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    domain = config["domain"]
    resolution = config["preprocessing"]["target_resolution"]
    latitude = np.arange(domain["south"], domain["north"] + resolution / 2, resolution)
    longitude = np.arange(domain["west"], domain["east"] + resolution / 2, resolution)
    return latitude, longitude


def _regrid_daily(
    data: xr.DataArray,
    latitude: np.ndarray,
    longitude: np.ndarray,
) -> xr.DataArray:
    regridded = data.interp(lat=latitude, lon=longitude, method="linear")
    return regridded.resample(time="1D").mean()


def _remove_generated_store(path: Path) -> None:
    if not path.exists():
        return
    try:
        os.system(f'attrib -R "{path}\\*" /S /D >NUL 2>NUL')
    except OSError:
        pass
    try:
        os.chmod(path, 0o777)
    except OSError:
        pass
    for child in path.rglob("*"):
        try:
            os.chmod(child, 0o777 if child.is_dir() else 0o666)
        except OSError:
            pass
    try:
        shutil.rmtree(path)
    except PermissionError as error:
        raise PermissionError(
            f"Cannot replace generated Zarr store {path}. Close programs using it and retry."
        ) from error


def harmonize(config_path: str | Path = "config.yaml") -> Path:
    """Create and save the common daily OceanEmbed Zarr dataset, processed one month at a time."""
    config = load_config(config_path)
    latitude, longitude = _target_coordinates(config)
    inputs = config["preprocessing"]["inputs"]

    # Partition all raw input files by month
    input_files_by_month: dict[str, dict[str, list[Path]]] = {}
    for name, input_config in inputs.items():
        files = _files_for_input(config, input_config)
        input_files_by_month[name] = _group_files_by_month(files)

    # Collect all unique months sorted chronologically
    all_months = sorted(
        {month for grouped in input_files_by_month.values() for month in grouped.keys()}
    )
    if not all_months:
        raise ValueError("No monthly partitions found for the configured inputs")

    output_path = Path(config["paths"]["processed_data_dir"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _remove_generated_store(output_path)

    total_months = len(all_months)
    print(f"Harmonizing {total_months} months one-at-a-time into {output_path}...", flush=True)

    for idx, month in enumerate(all_months):
        print(f"[{idx + 1}/{total_months}] Processing surface month {month}...", flush=True)
        month_harmonized: dict[str, xr.DataArray] = {}

        for name, input_config in inputs.items():
            month_files = input_files_by_month[name].get(month, [])
            if not month_files:
                raise FileNotFoundError(f"Missing files for variable {name} in month {month}")

            # Open files with Dask chunking so nothing is fully materialized until needed
            month_arrays = []
            for path in month_files:
                with xr.open_dataset(path, chunks={"time": 1}) as dataset:
                    month_arrays.append(_standardize_dataarray(dataset, input_config))

            month_da = xr.concat(month_arrays, dim="time").sortby("time")
            # Regrid down to target 0.25deg grid and daily mean, then compute to small in-memory array (~3 MB)
            regridded = _regrid_daily(month_da, latitude, longitude).compute()
            month_harmonized[name] = regridded

        reference = month_harmonized.get("sst", next(iter(month_harmonized.values())))
        for name, data in month_harmonized.items():
            if not np.allclose(data.lat.values, reference.lat.values):
                raise AssertionError(f"Latitude coordinates differ for {name} in month {month}")
            if not np.allclose(data.lon.values, reference.lon.values):
                raise AssertionError(f"Longitude coordinates differ for {name} in month {month}")

        month_combined = xr.Dataset(month_harmonized).sortby("time")
        valid_by_variable = xr.concat(
            [month_combined[name].notnull() for name in month_harmonized], dim="source"
        )
        month_ocean_mask = valid_by_variable.all(dim="source")
        month_combined = month_combined.where(month_ocean_mask)
        month_combined["ocean_mask"] = month_ocean_mask
        month_combined.attrs["description"] = "OceanEmbed daily harmonized source fields"

        # Write first month or append subsequent months along time dimension
        if idx == 0:
            month_combined.to_zarr(output_path, mode="w")
        else:
            month_combined.to_zarr(output_path, mode="a", append_dim="time")

        # Explicitly clear that month's data before moving to the next month
        del month_combined, month_harmonized, valid_by_variable, month_ocean_mask
        gc.collect()

    print(f"Successfully created {output_path} across {total_months} months", flush=True)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    harmonize(parser.parse_args().config)