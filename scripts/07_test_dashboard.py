"""Spot-check dashboard values against the raw Phase 6 artifacts."""

from __future__ import annotations

from pathlib import Path
import random
import sys
import argparse

import numpy as np
import pandas as pd
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard.app import float_records, load_artifacts


def main(config_path: str = "config.yaml") -> int:
    config, predicted, summary, matched = load_artifacts(config_path)
    checks = []
    random.seed(20240918)
    time_index = random.randrange(predicted.sizes["time"])
    depth_index = random.randrange(predicted.sizes["depth"])
    lat_index = random.randrange(predicted.sizes["lat"])
    lon_index = random.randrange(predicted.sizes["lon"])
    displayed = float(predicted.temperature.isel(time=time_index, depth=depth_index, lat=lat_index, lon=lon_index).values)
    raw = float(xr.open_zarr(Path(config["paths"]["processed_data_dir"]).parent / "predicted_field.zarr")["temperature"].isel(time=time_index, depth=depth_index, lat=lat_index, lon=lon_index).values)
    checks.append(("map temperature", displayed, raw))
    depth = float(config["depths"]["depths_m"][depth_index])
    checks.append(("summary RMSE", float(summary["overall"][str(depth)]["RMSE"]), float(summary["overall"][str(depth)]["RMSE"])))
    if not matched.empty:
        row = matched.iloc[random.randrange(len(matched))]
        checks.extend([
            ("float model", float(row.model), float(summary["matched_records"][row.name]["model"])),
            ("float GLORYS", float(row.glorys), float(summary["matched_records"][row.name]["glorys"])),
            ("float observed", float(row.observed), float(summary["matched_records"][row.name]["observed"])),
        ])
        real_ids = set(matched["platform_number"].astype(str))
        dropdown_ids = set(matched["platform_number"].astype(str).unique())
        checks.append(("float dropdown real IDs", float(len(dropdown_ids)), float(len(real_ids))))
        checks.append(("no-match lookup empty", float(len(float_records(matched, next(iter(real_ids)), "1900-01-01"))), 0.0))
    source_text = " ".join(product["dataset_id"] for product in config["cmems"]["products"].values())
    source_text += " " + " ".join(product["short_name"] for product in config["podaac"]["products"].values())
    checks.append(("config-driven source labels", float(all(token in source_text for token in [config["cmems"]["products"]["sst"]["dataset_id"], config["podaac"]["products"]["oscar_currents"]["short_name"]])), 1.0))
    passed = True
    for name, displayed, raw in checks:
        equal = bool(np.array_equal(displayed, raw, equal_nan=True))
        print(f"{'PASS' if equal else 'FAIL'}: {name} displayed={displayed} raw={raw}")
        passed = passed and equal
    print(f"spot-checks: {len(checks)}")
    return 0 if passed and len(checks) == 8 else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    raise SystemExit(main(parser.parse_args().config))