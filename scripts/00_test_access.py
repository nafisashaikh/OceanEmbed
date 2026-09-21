"""Download and plot one configured day and tile of the configured SST product."""

from __future__ import annotations

from pathlib import Path
import sys
import argparse

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.download.download_cmems import _login, _subset_product, _chunk_output_filename
import pandas as pd


def main(config_path: str | Path = "config.yaml") -> None:
    config = load_config(config_path)
    quick_test = config["quick_test"]
    product = config["cmems"]["products"][quick_test["product"]]
    domain = quick_test["domain"]
    test_config = {
        **config,
        "domain": domain,
        "dates": {
            "start": f"{quick_test['date']}T00:00:00Z",
            "end": f"{quick_test['date']}T23:59:59Z",
        },
    }

    output_directory = Path(config["paths"]["sanity_output"])
    marine = _import_copernicusmarine()
    _login(marine)
    _subset_product(marine, product, test_config, output_directory)

    month_label = pd.Timestamp(quick_test["date"]).strftime("%Y-%m")
    chunk_file = _chunk_output_filename(product["output_filename"], month_label)
    dataset_path = output_directory / chunk_file
    if not dataset_path.exists():
        dataset_path = output_directory / product["output_filename"]
    with xr.open_dataset(dataset_path) as dataset:
        print(dataset)
        print("shape:", dataset.sizes)
        print("coordinates:", list(dataset.coords))
        print("time range:", dataset["time"].min().item(), dataset["time"].max().item())
        variable = product["variables"][0]
        axis = dataset[variable].isel(time=0) if "time" in dataset[variable].dims else dataset[variable]
        figure = plt.figure(figsize=(8, 6))
        axis.plot(ax=plt.axes(projection=ccrs.PlateCarree()), transform=ccrs.PlateCarree())
        plt.gca().coastlines()
        figure.savefig(output_directory / quick_test["output_filename"], dpi=150, bbox_inches="tight")
        plt.close(figure)


def _import_copernicusmarine():
    import copernicusmarine

    return copernicusmarine


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    main(parser.parse_args().config)
