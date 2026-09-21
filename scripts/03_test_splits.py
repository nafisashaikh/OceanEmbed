"""Test split boundaries, train-only normalization, and dataset tensor shapes."""

from __future__ import annotations

from pathlib import Path
import sys
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data.dataset import INPUT_VARIABLES, OceanEmbedDataset, compute_normalization_stats, split_cubes


def _report(name: str, passed: bool, detail: str = "") -> bool:
    print(f"{'PASS' if passed else 'FAIL'}: {name}{(': ' + detail) if detail else ''}")
    return passed


def main(config_path: str | Path = "config.yaml") -> int:
    config = load_config(config_path)
    stats_path = compute_normalization_stats(config_path)
    splits = split_cubes(config_path)
    split_dates = {name: set(pd.DatetimeIndex(surface.time.values)) for name, (surface, _) in splits.items()}
    overlap = any(split_dates[a] & split_dates[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test")))
    checks = [_report("zero date overlap between splits", not overlap)]
    print("days per split:", {name: len(dates) for name, dates in split_dates.items()})
    print("normalization stats:", stats_path)

    train = OceanEmbedDataset("train", config_path)
    val = OceanEmbedDataset("val", config_path)
    test = OceanEmbedDataset("test", config_path)
    sample_input, sample_target = train[0]
    expected_channels = len(INPUT_VARIABLES) * (config["time"]["lag_days"] + 1)
    shapes_ok = sample_input.shape[0] == expected_channels and sample_target.shape[0] == len(config["depths"]["depths_m"])
    checks.append(_report("dataset tensor shapes", shapes_ok, f"input={tuple(sample_input.shape)}, target={tuple(sample_target.shape)}"))
    stats_reused = train.stats == val.stats == test.stats
    checks.append(_report("val/test reuse train-only normalization statistics", stats_reused))

    tensor_values = []
    for dataset in (train, val, test):
        for index in range(len(dataset)):
            input_tensor, target_tensor = dataset[index]
            tensor_values.extend((input_tensor, target_tensor))
    nan_free = all(bool(torch.isfinite(value).all()) for value in tensor_values)
    checks.append(_report("dataset tensors contain no NaN or infinite values", nan_free, "masked/missing cells are filled with zero"))

    raw_surface, _ = splits["train"]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    before = raw_surface[INPUT_VARIABLES[0]].values.ravel()
    before = before[np.isfinite(before)]
    train_mean = float(before.mean())
    train_std = float(before.std())
    normalized = (before - train_mean) / (train_std if train_std > 0 else 1.0)
    histogram_ok = abs(float(normalized.mean())) < 0.05 and 0.8 < float(normalized.std()) < 1.2
    axes[0].hist(before, bins=40)
    axes[0].set_title(f"Before: {INPUT_VARIABLES[0]}")
    axes[1].hist(normalized, bins=40)
    axes[1].set_title(f"After: normalized {INPUT_VARIABLES[0]}")
    figure.tight_layout()
    figure.savefig(Path(config["paths"]["sanity_output"]).parent / "input_normalization_histograms.png", dpi=150)
    plt.close(figure)
    checks.append(_report("normalized input histogram is centered and unit-ish", histogram_ok, f"mean={normalized.mean():.4f}, std={normalized.std():.4f}"))
    print(f"split dataset lengths: train={len(train)}, val={len(val)}, test={len(test)}")
    for name, dataset in (("train", train), ("val", val), ("test", test)):
        finite_days = sum(np.isfinite(dataset.surface[INPUT_VARIABLES[0]].values).any(axis=(1, 2)))
        print(f"finite {INPUT_VARIABLES[0]} days in {name}: {finite_days}/{len(dataset)}")
    return 0 if all(checks) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    raise SystemExit(main(parser.parse_args().config))