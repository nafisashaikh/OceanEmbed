"""Configuration loading and validation for OceanEmbed."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml


_REQUIRED_FIELDS = {
    "project": {"name"},
    "paths": {
        "data_root",
        "cmems_output",
        "podaac_output",
        "sanity_output",
        "raw_data_dir",
        "processed_data_dir",
        "model_checkpoint_dir",
    },
    "dates": {"start", "end"},
    "time": {"train_end", "val_end", "lag_days"},
    "domain": {"west", "east", "south", "north"},
    "depths": {"minimum", "maximum"},
    "argo": {
        "input_glob", "output_validation_table", "validation_split_ratio", "random_seed",
        "platform_field", "time_field", "latitude_field", "longitude_field",
        "pressure_field", "temperature_field",
    },
    "model": {
        "embedding_dim", "base_channels", "batch_size", "learning_rate", "epochs", "random_seed",
        "use_pretrained_encoder", "depth_weighting", "depth_weights"
    },
    "validation": {"spatial_tolerance_deg", "time_tolerance_days", "depth_tolerance_m", "regions"},
    "cmems": {"products"},
    "podaac": {"products"},
    "quick_test": {"product", "date", "domain", "output_filename"},
    "preprocessing": {"target_resolution", "inputs"},
}


def _require_mapping(config: dict[str, Any], section: str) -> dict[str, Any]:
    value = config.get(section)
    if not isinstance(value, dict):
        raise ValueError(f"Missing or invalid config section: {section}")
    return value


def _validate_required_fields(config: dict[str, Any]) -> None:
    for section, fields in _REQUIRED_FIELDS.items():
        values = _require_mapping(config, section)
        missing = sorted(field for field in fields if field not in values)
        if missing:
            raise KeyError(f"Missing required config fields in {section}: {', '.join(missing)}")

    domain = _require_mapping(config, "domain")
    quick_domain = _require_mapping(_require_mapping(config, "quick_test"), "domain")
    for name, values in (("domain", domain), ("quick_test.domain", quick_domain)):
        missing = sorted({"west", "east", "south", "north"} - values.keys())
        if missing:
            raise KeyError(f"Missing required config fields in {name}: {', '.join(missing)}")

    cmems_products = config["cmems"]["products"]
    if not isinstance(cmems_products, dict) or not cmems_products:
        raise ValueError("cmems.products must be a non-empty YAML mapping")
    for name, product in cmems_products.items():
        if not isinstance(product, dict):
            raise ValueError(f"cmems.products.{name} must be a YAML mapping")
        missing = sorted({"dataset_id", "variables", "output_filename", "output_subdirectory", "use_depth"} - product.keys())
        if missing:
            raise KeyError(f"Missing required config fields in cmems.products.{name}: {', '.join(missing)}")

    podaac_products = config["podaac"]["products"]
    if not isinstance(podaac_products, dict) or not podaac_products:
        raise ValueError("podaac.products must be a non-empty YAML mapping")
    for name, product in podaac_products.items():
        if not isinstance(product, dict):
            raise ValueError(f"podaac.products.{name} must be a YAML mapping")
        missing = sorted({"short_name", "version", "output_subdirectory"} - product.keys())
        if missing:
            raise KeyError(f"Missing required config fields in podaac.products.{name}: {', '.join(missing)}")

    quick_test = config["quick_test"]
    if quick_test["product"] not in cmems_products:
        raise KeyError(f"quick_test.product is not defined in cmems.products: {quick_test['product']}")

    preprocessing = config["preprocessing"]
    if preprocessing["target_resolution"] <= 0:
        raise ValueError("preprocessing.target_resolution must be positive")
    inputs = preprocessing["inputs"]
    required_inputs = {
        "sst", "sss", "ssh", "temperature", "current_u", "current_v", "wind_u", "wind_v"
    }
    missing = sorted(required_inputs - inputs.keys())
    if missing:
        raise KeyError(f"Missing preprocessing inputs: {', '.join(missing)}")
    for name, values in inputs.items():
        if not isinstance(values, dict):
            raise ValueError(f"preprocessing.inputs.{name} must be a YAML mapping")
        missing = sorted({"glob", "variable", "units"} - values.keys())
        if missing:
            raise KeyError(f"Missing required config fields in preprocessing.inputs.{name}: {', '.join(missing)}")


def _validate_ranges(config: dict[str, Any]) -> None:
    domain = config["domain"]
    if not (-180 <= domain["west"] < domain["east"] <= 180):
        raise ValueError("domain longitude bounds must satisfy -180 <= west < east <= 180")
    if not (-90 <= domain["south"] < domain["north"] <= 90):
        raise ValueError("domain latitude bounds must satisfy -90 <= south < north <= 90")

    depths = config["depths"]
    if depths["minimum"] < 0 or depths["minimum"] > depths["maximum"]:
        raise ValueError("depths must satisfy 0 <= minimum <= maximum")
    depths_m = depths.get("depths_m")
    if not isinstance(depths_m, list) or not depths_m or any(depth < 0 for depth in depths_m):
        raise ValueError("depths.depths_m must be a non-empty list of non-negative values")
    if depths_m != sorted(set(depths_m)):
        raise ValueError("depths.depths_m must be sorted and contain no duplicates")
    if not 0 < config["argo"]["validation_split_ratio"] < 1:
        raise ValueError("argo.validation_split_ratio must be between 0 and 1")
    start = pd.Timestamp(config["dates"]["start"])
    train_end = pd.Timestamp(config["time"]["train_end"])
    val_end = pd.Timestamp(config["time"]["val_end"])
    end = pd.Timestamp(config["dates"]["end"])
    if not start <= train_end < val_end < end:
        raise ValueError("time boundaries must satisfy dates.start <= train_end < val_end < dates.end")
    if config["time"]["lag_days"] < 0:
        raise ValueError("time.lag_days must be non-negative")
    model = config["model"]
    if model["embedding_dim"] <= 0 or model["base_channels"] <= 0:
        raise ValueError("model embedding_dim and base_channels must be positive")
    if model["batch_size"] <= 0 or model["learning_rate"] <= 0 or model["epochs"] <= 0:
        raise ValueError("model batch_size, learning_rate, and epochs must be positive")
    if model["depth_weighting"] not in {"uniform", "inverse_std", "explicit"}:
        raise ValueError("model.depth_weighting must be uniform, inverse_std, or explicit")
    if model["depth_weighting"] == "explicit" and len(model["depth_weights"]) != len(config["depths"]["depths_m"]):
        raise ValueError("model.depth_weights must have one value per configured depth")
    validation = config["validation"]
    if validation["spatial_tolerance_deg"] <= 0 or validation["time_tolerance_days"] <= 0 or validation["depth_tolerance_m"] <= 0:
        raise ValueError("validation tolerances must be positive")
    for region_name, region in validation["regions"].items():
        if not isinstance(region, dict) or not (-180 <= region["west"] < region["east"] <= 180 and -90 <= region["south"] < region["north"] <= 90):
            raise ValueError(f"validation.regions.{region_name} bounds are invalid")

    quick_domain = config["quick_test"]["domain"]
    if not (-180 <= quick_domain["west"] < quick_domain["east"] <= 180):
        raise ValueError("quick_test.domain longitude bounds are invalid")
    if not (-90 <= quick_domain["south"] < quick_domain["north"] <= 90):
        raise ValueError("quick_test.domain latitude bounds are invalid")


def load_config(config_path: str | Path = "config.yaml") -> dict[str, Any]:
    """Load and validate one OceanEmbed YAML configuration file."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")

    with path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a YAML mapping")
    _validate_required_fields(config)
    _validate_ranges(config)
    return config
