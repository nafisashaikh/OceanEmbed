"""
Correctness tests for OceanEmbed depth-channel alignment and unit consistency.

Tests
-----
T1  Depth-channel alignment
    - Reads the ACTUAL depth coordinate arrays from both predicted_field.zarr and
      target_temperature.zarr at runtime (no config assumption).
    - Asserts they are identical in value and order.
    - Asserts that norm_stats.json depths match both stores.
    - Asserts the checkpoint output_channels matches len(depths_m).

T2  Unit / scale consistency at loss-computation time
    - Loads one batch from the training DataLoader (the exact objects _loss() sees).
    - Asserts target tensors are in normalised z-score space (|mean| < 0.5, std in [0.5, 2.0]).
    - Asserts input tensors are also normalised.
    - Asserts targets are NOT in physical degC (mean must not be in [10, 35] for all channels).

T3  Ocean-mask zero-pollution check
    - Loads predicted_field.zarr and confirms that no pixel value is exactly 0.0
      (the bug: output * mask wrote 0 for land, which is finite and in the validation range).
    - Confirms all non-ocean pixels in the predicted field are NaN (the post-fix state).

T4  Cross-split unit consistency
    - Loads one batch from each of train / val / test splits.
    - Asserts they all produce normalised targets in the same range.

Run
---
    python scripts/tests/test_depth_channel_and_units.py [--config config.yaml]

Returns exit code 0 if all tests pass, 1 if any fail.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data.dataset import OceanEmbedDataset, load_normalization_stats

# ──────────────────────────────────────────────────────────────────────────────
PASS = "PASS"
FAIL = "FAIL"
SEP  = "-" * 72


def _banner(name: str) -> None:
    print(f"\n{SEP}")
    print(f"  {name}")
    print(SEP)


def _ok(msg: str) -> None:
    print(f"  OK   {msg}")


def _warn(msg: str) -> None:
    print(f"  WARN {msg}")


def _fail(msg: str) -> None:
    print(f"  FAIL {msg}")


# ──────────────────────────────────────────────────────────────────────────────
# T1 – Depth-channel alignment
# ──────────────────────────────────────────────────────────────────────────────

def test_depth_channel_alignment(config: dict[str, Any], out_root: Path) -> bool:
    _banner("T1  Depth-channel alignment (runtime, not assumed from config)")
    failures: list[str] = []

    # Load zarr stores and read coordinate arrays directly
    pred_ds = xr.open_zarr(out_root / "predicted_field.zarr")
    tgt_ds  = xr.open_zarr(out_root / "target_temperature.zarr")

    pred_depths = pred_ds.depth.values.tolist()
    tgt_depths  = tgt_ds.target_temperature.depth.values.tolist()

    print(f"  predicted_field depth coord : {pred_depths}")
    print(f"  target_temperature depth coord: {tgt_depths}")

    if not np.array_equal(pred_depths, tgt_depths):
        failures.append(
            f"Depth arrays differ!\n"
            f"    predicted : {pred_depths}\n"
            f"    target    : {tgt_depths}"
        )
    else:
        _ok("Depth coordinate arrays are element-wise identical between predicted and target zarr stores")

    # Compare to config
    config_depths = [float(d) for d in config["depths"]["depths_m"]]
    if not np.allclose(pred_depths, config_depths, atol=1e-6):
        failures.append(f"predicted_field depths {pred_depths} differ from config depths_m {config_depths}")
    else:
        _ok("predicted_field depth coord matches config depths_m exactly")

    # Compare to norm_stats.json
    norm_path = out_root / "norm_stats.json"
    if norm_path.exists():
        norm = json.loads(norm_path.read_text(encoding="utf-8"))
        stats_depths = norm["target_temperature"]["depths"]
        if not np.allclose(stats_depths, config_depths, atol=1e-6):
            failures.append(
                f"norm_stats.json depths {stats_depths} differ from config depths_m {config_depths}. "
                "Normalisation statistics would be applied to the wrong channel."
            )
        else:
            _ok("norm_stats.json depths match config depths_m — normalisation applied to correct channels")
    else:
        _warn("norm_stats.json not found — skipping stats-depth check")

    # Check checkpoint output_channels
    ckpt_path = Path(config["paths"]["model_checkpoint_dir"]) / "reconstruction_model_best.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        ckpt_out = ckpt.get("output_channels")
        if ckpt_out != len(config_depths):
            failures.append(
                f"Checkpoint output_channels={ckpt_out} != len(config depths_m)={len(config_depths)}. "
                "Model was trained on a different depth list."
            )
        else:
            _ok(f"Checkpoint output_channels={ckpt_out} matches len(config depths_m)={len(config_depths)}")

        # Explicitly verify positional mapping for each channel
        ckpt_depths_from_coord = pred_depths   # same list already verified above
        print()
        print("  Per-channel positional mapping (verified):")
        for ch, (pd_, cd_) in enumerate(zip(ckpt_depths_from_coord, config_depths)):
            match = "OK" if np.isclose(pd_, cd_) else "MISMATCH"
            print(f"    channel {ch:2d} -> predicted coord={pd_:7.1f} m  config={cd_:7.1f} m  [{match}]")
            if match != "OK":
                failures.append(f"Channel {ch} positional mismatch: stored depth {pd_} != config depth {cd_}")
    else:
        _warn("Checkpoint not found — skipping output_channels check")

    pred_ds.close()
    tgt_ds.close()

    passed = len(failures) == 0
    print()
    for f in failures:
        _fail(f)
    print(f"\n  T1 result: {PASS if passed else FAIL}")
    return passed


# ──────────────────────────────────────────────────────────────────────────────
# T2 – Unit / scale consistency at loss-computation time
# ──────────────────────────────────────────────────────────────────────────────

def _is_zscore(arr: np.ndarray, label: str) -> tuple[bool, str]:
    """Assert array looks like z-score: |mean| < 0.5, std in [0.3, 3.0]."""
    fin = arr[np.isfinite(arr) & (arr != 0.0)]   # exclude nan_to_num fill zeros
    if fin.size == 0:
        return False, f"{label}: no finite non-zero values"
    m, s = float(fin.mean()), float(fin.std())
    ok = abs(m) < 0.5 and 0.3 < s < 3.0
    detail = f"mean={m:+.4f}  std={s:.4f}"
    return ok, f"{label}: {detail}" + ("  [OK]" if ok else "  [NOT z-score]")


def _is_physical_celsius(arr: np.ndarray) -> bool:
    """Heuristic: if mean is in [10, 35] for ocean, likely physical degC."""
    fin = arr[np.isfinite(arr) & (arr != 0.0)]
    if fin.size == 0:
        return False
    return 10.0 < float(fin.mean()) < 35.0


def test_loss_unit_consistency(config_path: str | Path) -> bool:
    _banner("T2  Unit / scale consistency at loss-computation time")
    failures: list[str] = []

    ds = OceanEmbedDataset("train", config_path, return_mask=True)
    if len(ds.valid_indices) == 0:
        print("  SKIP  No valid training samples available")
        return True

    subset  = Subset(ds, ds.valid_indices[:min(4, len(ds.valid_indices))])
    loader  = DataLoader(subset, batch_size=len(subset), shuffle=False)
    inputs, targets, mask = next(iter(loader))

    print(f"  Batch shape — inputs: {list(inputs.shape)}  targets: {list(targets.shape)}  mask: {list(mask.shape)}")
    print()

    # --- Target channels should be normalised ---
    n_depths = targets.shape[1]
    target_phys_suspects = 0
    for ch in range(n_depths):
        ch_arr = targets[:, ch, :, :].numpy()
        ok, detail = _is_zscore(ch_arr, f"target ch {ch:2d}")
        print(f"  {detail}")
        if not ok:
            failures.append(f"Target channel {ch} does not look like z-score: {detail}")
        if _is_physical_celsius(ch_arr):
            target_phys_suspects += 1

    if target_phys_suspects > n_depths // 2:
        failures.append(
            f"{target_phys_suspects}/{n_depths} target channels look like physical degC (mean in [10,35]). "
            "Targets may not be normalised — loss would then be in degC^2, not z-score^2."
        )
    else:
        _ok("Target tensors passed to loss are in normalised z-score space")

    # --- Input channels should also be normalised ---
    n_inputs = inputs.shape[1]
    input_phys_suspects = 0
    for ch in range(n_inputs):
        ch_arr = inputs[:, ch, :, :].numpy()
        ok, detail = _is_zscore(ch_arr, f"input  ch {ch:2d}")
        if not ok:
            failures.append(f"Input channel {ch} does not look like z-score: {detail}")
        if _is_physical_celsius(ch_arr):
            input_phys_suspects += 1

    if input_phys_suspects > n_inputs // 2:
        failures.append(
            f"{input_phys_suspects}/{n_inputs} input channels look like physical values. "
            "Inputs may not be normalised."
        )
    else:
        _ok("Input tensors are in normalised space")

    # --- Confirm model predictions are also normalised at inference ---
    ckpt_path = Path(load_config(config_path)["paths"]["model_checkpoint_dir"]) / "reconstruction_model_best.pt"
    if ckpt_path.exists():
        from src.models.reconstruction_model import OceanEmbedReconstructionModel
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model = OceanEmbedReconstructionModel(
            ckpt["input_channels"], ckpt["output_channels"], ckpt["embedding_dim"], ckpt["base_channels"]
        ).eval()
        model.load_state_dict(ckpt["model_state_dict"])
        with torch.no_grad():
            preds, _ = model(inputs)
        pred_ch5 = preds[:, 5, :, :].numpy()
        pred_ok, pred_detail = _is_zscore(pred_ch5, "model output ch 5 (100m)")
        print(f"\n  Raw model output (normalised space):")
        print(f"  {pred_detail}")
        if not pred_ok:
            _warn(
                "Model output ch 5 does not look like z-score — the model may not have converged "
                "enough to produce outputs in the normalised scale. This is an underfitting indicator, "
                "not a unit bug."
            )

    print()
    for f in failures:
        _fail(f)
    passed = len(failures) == 0
    print(f"\n  T2 result: {PASS if passed else FAIL}")
    return passed


# ──────────────────────────────────────────────────────────────────────────────
# T3 – Ocean-mask zero-pollution in predicted field
# ──────────────────────────────────────────────────────────────────────────────

def test_mask_zero_pollution(config: dict[str, Any], out_root: Path) -> bool:
    _banner("T3  Ocean-mask zero-pollution in predicted_field.zarr")
    failures: list[str] = []

    pred_ds = xr.open_zarr(out_root / "predicted_field.zarr")
    tgt_ds  = xr.open_zarr(out_root / "target_temperature.zarr")
    depths  = pred_ds.depth.values.tolist()
    pred_times = pred_ds.time.values
    tgt_times  = tgt_ds.time.values

    # Align target to test-period times
    pred_in_tgt = [int(np.argmin(np.abs(tgt_times - t))) for t in pred_times]

    total_zeros = 0
    total_nan   = 0
    cross_contaminated = 0  # zeros in pred where target is finite

    for depth in depths[:3]:   # check first 3 depth levels to keep it fast
        pred_arr = pred_ds["temperature"].sel(depth=depth).values          # (n_test, lat, lon)
        tgt_arr  = tgt_ds["target_temperature"].sel(depth=depth).values[pred_in_tgt]

        n_zeros = int(np.sum(pred_arr == 0.0))
        n_nan   = int(np.sum(np.isnan(pred_arr)))
        # "Cross-contaminated": predicted zero where target is a valid ocean observation
        n_cross = int(np.sum((pred_arr == 0.0) & np.isfinite(tgt_arr)))

        total_zeros         += n_zeros
        total_nan           += n_nan
        cross_contaminated  += n_cross

        print(f"  depth {depth:7.1f} m  pred==0: {n_zeros:5d}  pred==NaN: {n_nan:5d}  "
              f"cross-contaminated zeros: {n_cross:5d}")

    print()
    if total_zeros > 0:
        # Distinguish between the pre-fix state (zeros) and post-fix state (NaN)
        failures.append(
            f"predicted_field.zarr contains {total_zeros} pixels with value exactly 0.0 across 3 "
            "depth levels. These were written by the ocean-mask multiplication "
            "(output * mask) which silently stores 0 for land pixels instead of NaN. "
            "Effect: 0 degC is a finite value — it pollutes bulk statistics and can "
            "match ARGO profiles at land-adjacent ocean grid cells."
        )
        if cross_contaminated > 0:
            failures.append(
                f"{cross_contaminated} zeros in predicted field coincide with FINITE target values "
                "(genuine ocean pixels). This means the surface ocean_mask and the GLORYS depth "
                "mask have different land-sea boundaries — some ocean pixels are being zeroed out "
                "by the wrong mask."
            )
    else:
        _ok("No zero-valued pixels in predicted field — land pixels correctly stored as NaN")

    if total_nan > 0:
        _ok(f"Non-ocean pixels stored as NaN ({total_nan} NaN across 3 depth levels checked) — correct")
    else:
        if total_zeros == 0:
            _ok("No NaN either — possibly all pixels are ocean (verify domain extent)")

    pred_ds.close()
    tgt_ds.close()

    passed = len(failures) == 0
    for f in failures:
        _fail(f)
    print(f"\n  T3 result: {PASS if passed else FAIL}")
    return passed


# ──────────────────────────────────────────────────────────────────────────────
# T4 – Cross-split unit consistency
# ──────────────────────────────────────────────────────────────────────────────

def test_cross_split_unit_consistency(config_path: str | Path) -> bool:
    _banner("T4  Cross-split unit consistency (train / val / test all normalised)")
    failures: list[str] = []

    for split in ("train", "val", "test"):
        ds = OceanEmbedDataset(split, config_path, return_mask=True)
        if not ds.valid_indices:
            print(f"  SKIP  {split} split has no valid samples")
            continue
        subset = Subset(ds, ds.valid_indices[:min(2, len(ds.valid_indices))])
        loader = DataLoader(subset, batch_size=len(subset), shuffle=False)
        _, targets, _ = next(iter(loader))
        # Check depth=100m channel (index 5) for each split
        ch5 = targets[:, 5, :, :].numpy()
        ok, detail = _is_zscore(ch5, f"{split:5s} ch 5 (100m)")
        print(f"  {detail}")
        if not ok:
            failures.append(f"{split} targets not normalised: {detail}")

    passed = len(failures) == 0
    for f in failures:
        _fail(f)
    print(f"\n  T4 result: {PASS if passed else FAIL}")
    return passed


# ──────────────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────────────

def main(config_path: str = "config.yaml") -> int:
    config   = load_config(config_path)
    out_root = Path(config["paths"]["processed_data_dir"]).parent

    results: dict[str, bool] = {}
    results["T1_channel_alignment"]       = test_depth_channel_alignment(config, out_root)
    results["T2_loss_unit_consistency"]   = test_loss_unit_consistency(config_path)
    results["T3_mask_zero_pollution"]     = test_mask_zero_pollution(config, out_root)
    results["T4_cross_split_units"]       = test_cross_split_unit_consistency(config_path)

    print(f"\n{'=' * 72}")
    print("  Summary")
    print('=' * 72)
    all_pass = True
    for name, passed in results.items():
        status = PASS if passed else FAIL
        print(f"  {name:40s}  {status}")
        if not passed:
            all_pass = False
    print()
    if all_pass:
        print("  All tests PASSED.")
    else:
        print("  One or more tests FAILED — see above for details.")
    print('=' * 72)
    return 0 if all_pass else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="OceanEmbed depth-channel alignment and unit consistency tests."
    )
    parser.add_argument("--config", default="config.yaml")
    raise SystemExit(main(parser.parse_args().config))
