import copy
import json

import numpy as np
import pandas as pd
import yaml
import xarray as xr

import src.data.dataset as dataset_module
from src.data.dataset import _split_masks, compute_normalization_stats
from src.config import load_config


def test_date_split_masks_are_disjoint():
    config = {"time": {"train_end": "2024-01-02T00:00:00Z", "val_end": "2024-01-03T00:00:00Z"}}
    masks = _split_masks(pd.date_range("2024-01-01", periods=4), config)
    assert sum(mask.sum() for mask in masks.values()) == 4
    assert not np.any(masks["train"] & masks["val"])
    assert not np.any(masks["train"] & masks["test"])
    assert not np.any(masks["val"] & masks["test"])


def test_normalization_stats_use_training_values_only(tmp_path, monkeypatch):
    config = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    config["paths"]["processed_data_dir"] = str(tmp_path / "surface.zarr")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    variables = dataset_module.INPUT_VARIABLES
    surface = xr.Dataset({name: (("time", "lat", "lon"), np.full((2, 1, 1), 1.0 + index)) for index, name in enumerate(variables)})
    target = xr.Dataset({"target_temperature": (("time", "depth", "lat", "lon"), np.ones((2, 15, 1, 1)))}, coords={"depth": config["depths"]["depths_m"]})
    monkeypatch.setattr(dataset_module, "split_cubes", lambda _: {"train": (surface, target), "val": (surface, target), "test": (surface, target)})
    output = compute_normalization_stats(config_path)
    stats = json.loads(output.read_text(encoding="utf-8"))
    assert stats["inputs"]["sst"]["mean"] == 1.0
    assert stats["inputs"]["wind_v"]["mean"] == 8.0
    assert stats["target_temperature"]["mean"] == [1.0] * 15
