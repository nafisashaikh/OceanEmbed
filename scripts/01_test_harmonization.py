"""Validate and plot the harmonized OceanEmbed Zarr dataset."""

from __future__ import annotations

from pathlib import Path
import sys
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import pandas as pd
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.preprocessing.harmonize import harmonize


def _report(name: str, passed: bool, detail: str = "") -> bool:
    print(f"{'PASS' if passed else 'FAIL'}: {name}{(': ' + detail) if detail else ''}")
    return passed


def _plot_variables(dataset: xr.Dataset, output_dir: Path) -> None:
    for name in dataset.data_vars:
        if name == "ocean_mask":
            continue
        figure = plt.figure(figsize=(8, 6))
        axis = figure.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        dataset[name].isel(time=0).plot(ax=axis, transform=ccrs.PlateCarree(), add_colorbar=True)
        axis.coastlines()
        axis.set_title(f"{name} - {pd.Timestamp(dataset.time.values[0]).date()}")
        figure.savefig(output_dir / f"harmonized_{name}.png", dpi=150, bbox_inches="tight")
        plt.close(figure)


def main(config_path: str | Path = "config.yaml") -> int:
    config = load_config(config_path)
    store = Path(config["paths"]["processed_data_dir"])
    if not store.exists():
        harmonize(config_path)
    dataset = xr.open_zarr(store)
    variables = [name for name in dataset.data_vars if name != "ocean_mask"]
    reference = dataset[variables[0]]
    checks = []
    checks.append(_report(
        "all variables share identical lat/lon coordinate values",
        all(np.allclose(dataset[name].lat, reference.lat) and np.allclose(dataset[name].lon, reference.lon) for name in variables),
    ))

    start_date = pd.Timestamp(config["dates"]["start"])
    end_date = pd.Timestamp(config["dates"]["end"])
    if start_date.tzinfo is not None:
        start_date = start_date.tz_convert(None)
    if end_date.tzinfo is not None:
        end_date = end_date.tz_convert(None)
    expected = pd.date_range(
        start_date.floor("D"),
        end_date.floor("D"),
        freq="D",
    )
    actual = pd.DatetimeIndex(dataset.time.values)
    if actual.tz is not None:
        actual = actual.tz_convert(None)
    gaps = expected.difference(actual)
    checks.append(_report("no unexpected date gaps in the time index", gaps.empty, f"gaps={list(gaps)}" if len(gaps) else ""))

    ranges = {"sst": (-2, 38), "sss": (20, 45), "ssh": (-1.5, 1.5)}
    for name, (minimum, maximum) in ranges.items():
        values = dataset[name].values
        finite = values[np.isfinite(values)]
        passed = finite.size > 0 and finite.min() >= minimum and finite.max() <= maximum
        checks.append(_report(name + " physically plausible range", passed, f"range={finite.min(), finite.max()}" if finite.size else "no finite values"))

    mask = dataset["ocean_mask"].values
    consistent = all(np.all(np.isnan(dataset[name].values[~mask])) for name in variables)
    checks.append(_report("land pixels are NaN/masked consistently across every variable", consistent))

    output_dir = Path(config["paths"]["sanity_output"]).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    _plot_variables(dataset, output_dir)
    dataset.close()
    return 0 if all(checks) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    raise SystemExit(main(parser.parse_args().config))