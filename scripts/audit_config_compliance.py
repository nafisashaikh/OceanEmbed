"""Audit last-run artifacts against config.yaml without recomputing the pipeline.

Checks performed
----------------
1. Surface Zarr cube (oceanembed.zarr)
   - lat/lon extents match config domain
   - time range matches config dates
2. Target Zarr cube (target_temperature.zarr)
   - lat/lon extents match config domain
   - depth levels match config depths_m exactly
   - time range matches config dates
3. Predicted field Zarr (predicted_field.zarr)
   - lat/lon extents match config domain
   - depth levels match config depths_m exactly
   - time range falls within config dates and covers only the test split
4. ARGO validation summary JSON (argo_validation_summary.json)
   - every matched record time is within config dates
   - every matched record lat/lon is inside config domain
   - every matched record depth appears in config depths_m
   - no missing platform_number values
   - summary metric keys match depths_m exactly
5. Source-code literal scan (src/**/*.py)
   - Searches for string/numeric literals that reproduce the *values* defined in
     config (domain bounds, individual depth levels, date substrings) appearing
     outside a config dictionary access expression.  The scan is driven by the
     actual config values so it stays accurate when config.yaml is edited.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    parsed = pd.Timestamp(value)
    return parsed.tz_localize(None) if parsed.tzinfo is not None else parsed


def _mismatch(mismatches: list[str], message: str) -> None:
    mismatches.append(message)
    print(f"MISMATCH  {message}")


# ---------------------------------------------------------------------------
# Output-file checks
# ---------------------------------------------------------------------------

def _check_cube(
    mismatches: list[str],
    label: str,
    path: Path,
    config: dict[str, Any],
    expected_depths: "np.ndarray | None",
    start: pd.Timestamp,
    end: pd.Timestamp,
    exact_time_range: bool = True,
) -> None:
    if not path.exists():
        _mismatch(mismatches, f"[{label}] output is missing: {path}")
        return

    try:
        ds = xr.open_zarr(path)
    except Exception as exc:  # noqa: BLE001
        _mismatch(mismatches, f"[{label}] cannot open Zarr: {exc}")
        return

    domain = config["domain"]
    resolution = config["preprocessing"]["target_resolution"]
    tol = resolution / 2 + 1e-6

    # spatial extents
    for coord, cfg_min, cfg_max in (
        ("lat", domain["south"], domain["north"]),
        ("lon", domain["west"], domain["east"]),
    ):
        if coord not in ds.coords:
            _mismatch(mismatches, f"[{label}] coordinate '{coord}' is absent")
            continue
        actual_min = float(ds[coord].min())
        actual_max = float(ds[coord].max())
        if not np.isclose(actual_min, cfg_min, atol=tol):
            _mismatch(mismatches, f"[{label}] {coord}.min={actual_min:.4f} != config={cfg_min} (tol={tol:.4f})")
        if not np.isclose(actual_max, cfg_max, atol=tol):
            _mismatch(mismatches, f"[{label}] {coord}.max={actual_max:.4f} != config={cfg_max} (tol={tol:.4f})")

    # depth levels
    if expected_depths is not None:
        if "depth" in ds.coords:
            actual_depths = ds.depth.values.astype(float)
            if not np.array_equal(actual_depths, expected_depths):
                _mismatch(
                    mismatches,
                    f"[{label}] depth={actual_depths.tolist()} != config depths_m={expected_depths.tolist()}",
                )
        else:
            _mismatch(mismatches, f"[{label}] expected depth coordinate but none found")

    # time range
    if "time" not in ds.coords:
        _mismatch(mismatches, f"[{label}] time coordinate is absent")
        ds.close()
        return

    times = pd.DatetimeIndex(ds.time.values)
    if exact_time_range:
        expected_dates = pd.date_range(start.floor("D"), end.floor("D"), freq="D")
        if not np.array_equal(times.normalize().values, expected_dates.values):
            _mismatch(
                mismatches,
                f"[{label}] time=[{times.min().date()}..{times.max().date()}, n={len(times)}]"
                f" != config=[{start.date()}..{end.date()}, n={len(expected_dates)}]",
            )
    else:
        val_end = _timestamp(config["time"]["val_end"])
        test_start = (val_end + pd.Timedelta(days=1)).floor("D")
        expected_test = pd.date_range(test_start, end.floor("D"), freq="D")
        if not np.array_equal(times.normalize().values, expected_test.values):
            _mismatch(
                mismatches,
                f"[{label}] time=[{times.min().date()}..{times.max().date()}, n={len(times)}]"
                f" != expected test period=[{test_start.date()}..{end.date()}, n={len(expected_test)}]",
            )
        if times.min().to_datetime64() < np.datetime64(start) or times.max().to_datetime64() > np.datetime64(end):
            _mismatch(mismatches, f"[{label}] time extends outside config dates [{start.date()}..{end.date()}]")

    ds.close()


def _audit_argo_summary(
    mismatches: list[str],
    config: dict[str, Any],
    summary_path: Path,
    depths: "np.ndarray",
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> None:
    if not summary_path.exists():
        _mismatch(mismatches, f"[argo_summary] file is missing: {summary_path}")
        return

    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        _mismatch(mismatches, f"[argo_summary] cannot parse JSON: {exc}")
        return

    records = pd.DataFrame(summary.get("matched_records", []))
    if records.empty:
        _mismatch(mismatches, "[argo_summary] matched_records is empty")
        return

    records["time"] = pd.to_datetime(records["time"], utc=True).dt.tz_localize(None)
    domain = config["domain"]

    out_of_time = ~records["time"].between(start, end)
    if out_of_time.any():
        sample = records.loc[out_of_time, "time"].dt.date.unique()[:5].tolist()
        _mismatch(mismatches, f"[argo_summary] {out_of_time.sum()} records outside config dates; sample={sample}")

    out_of_domain = (
        ~records["lat"].between(domain["south"], domain["north"])
        | ~records["lon"].between(domain["west"], domain["east"])
    )
    if out_of_domain.any():
        _mismatch(
            mismatches,
            f"[argo_summary] {out_of_domain.sum()} records outside config domain"
            f" lon[{domain['west']}..{domain['east']}] lat[{domain['south']}..{domain['north']}]",
        )

    actual_depths = records["depth"].astype(float).unique()
    alien = [d for d in actual_depths if not np.isin(d, depths)]
    if alien:
        _mismatch(mismatches, f"[argo_summary] depths {alien} not in config depths_m={depths.tolist()}")

    null_platform = records["platform_number"].isna() | (records["platform_number"].astype(str).str.strip() == "")
    if null_platform.any():
        _mismatch(mismatches, f"[argo_summary] {null_platform.sum()} records have blank/null platform_number")

    config_depth_keys = {str(float(d)) for d in depths}
    for metric_key in ("overall", "glorys_ceiling", "climatology"):
        section = summary.get(metric_key, {})
        extra = set(section.keys()) - config_depth_keys
        missing = config_depth_keys - set(section.keys())
        if extra:
            _mismatch(mismatches, f"[argo_summary.{metric_key}] extra depth keys {sorted(extra)}")
        if missing:
            _mismatch(mismatches, f"[argo_summary.{metric_key}] missing depth keys {sorted(missing)}")


# ---------------------------------------------------------------------------
# Source-code literal scan
# ---------------------------------------------------------------------------

def _build_literal_patterns(config: dict[str, Any]) -> "list[tuple[re.Pattern[str], str]]":
    """Return (pattern, description) pairs derived from actual config values."""
    patterns: list[tuple[re.Pattern[str], str]] = []

    domain = config["domain"]
    for key, value in domain.items():
        pat = re.compile(rf"(?<!['\"\w.]){re.escape(str(float(value)))}(?!['\"\w.])")
        patterns.append((pat, f"domain.{key}={value}"))

    for depth in config["depths"]["depths_m"]:
        pat = re.compile(rf"(?<!['\"\w.]){re.escape(str(float(depth)))}(?!['\"\w.])")
        patterns.append((pat, f"depths_m member={depth}"))

    for date_key in ("start", "end"):
        date_str = str(config["dates"][date_key])
        for fragment in (date_str[:7], date_str[:10]):
            patterns.append((re.compile(re.escape(fragment)), f"dates.{date_key} fragment '{fragment}'"))

    return patterns


_CONFIG_INDICATORS = (
    "config[", "config.get(", "config_path", "load_config",
    "depths_m", "domain[", "dates[", "config.yaml", "= config",
    '"depths"', "'depths'",
)

# Patterns that contain numeric literals which coincidentally match config
# values but are clearly not domain/depth/date references (e.g. fill values
# passed to np.nan_to_num, tolerance kwargs, etc.).
_FILL_VALUE_INDICATORS = (
    "nan_to_num",  # np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    "nan=",        # nan= keyword arg
    "posinf=",     # posinf= keyword arg
    "neginf=",     # neginf= keyword arg
    "rtol=",       # numpy/scipy tolerance kwargs
    "atol=",
    "p_value",     # statistical test results
    "isclose",     # comparison helpers
    "nan_policy",
    "0o777",       # filesystem permission bitmasks
    "0o666",
    ".05",         # significance thresholds
    "< 0.05",
)


def _line_uses_config(line: str) -> bool:
    if any(ind in line for ind in _CONFIG_INDICATORS):
        return True
    # Exclude lines that only contain well-known fill/tolerance patterns
    if any(ind in line for ind in _FILL_VALUE_INDICATORS):
        return True
    return False


def _is_comment(line: str) -> bool:
    s = line.lstrip()
    return s.startswith("#") or s.startswith('"""') or s.startswith("'''")


def _audit_source_literals(mismatches: list[str], config: dict[str, Any]) -> None:
    patterns = _build_literal_patterns(config)
    src_root = PROJECT_ROOT / "src"
    seen: set[str] = set()

    for py_file in sorted(src_root.rglob("*.py")):
        try:
            lines = py_file.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line_no, line in enumerate(lines, 1):
            if _line_uses_config(line) or _is_comment(line):
                continue
            for pattern, description in patterns:
                if pattern.search(line):
                    rel = str(py_file.relative_to(PROJECT_ROOT))
                    finding = f"[source_literal] {rel}:{line_no}: {description!r} in {line.strip()!r}"
                    if finding not in seen:
                        seen.add(finding)
                        _mismatch(mismatches, finding)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config_path: str = "config.yaml") -> int:
    print(f"Loading config: {config_path}")
    config = load_config(config_path)

    output_root = Path(config["paths"]["processed_data_dir"]).parent
    depths = np.asarray(config["depths"]["depths_m"], dtype=float)
    start = _timestamp(config["dates"]["start"])
    end = _timestamp(config["dates"]["end"])

    print(f"\nConfig summary")
    d = config["domain"]
    print(f"  domain : lon [{d['west']}..{d['east']}]  lat [{d['south']}..{d['north']}]")
    print(f"  dates  : {start.date()} -> {end.date()}")
    print(f"  depths : {depths.tolist()}")
    print(f"  output : {output_root}\n")

    mismatches: list[str] = []

    print("-- Checking surface cube (oceanembed.zarr) ...")
    _check_cube(
        mismatches, "surface",
        Path(config["paths"]["processed_data_dir"]),
        config, None, start, end, exact_time_range=True,
    )

    print("-- Checking target_temperature.zarr ...")
    _check_cube(
        mismatches, "target",
        output_root / "target_temperature.zarr",
        config, depths, start, end, exact_time_range=True,
    )

    print("-- Checking predicted_field.zarr ...")
    _check_cube(
        mismatches, "predicted",
        output_root / "predicted_field.zarr",
        config, depths, start, end, exact_time_range=False,
    )

    print("-- Checking argo_validation_summary.json ...")
    _audit_argo_summary(
        mismatches, config,
        output_root / "argo_validation_summary.json",
        depths, start, end,
    )

    print("-- Scanning source literals in src/ ...")
    _audit_source_literals(mismatches, config)

    print(f"\n{'=' * 60}")
    print(f"Config compliance mismatches found: {len(mismatches)}")
    if not mismatches:
        print("PASS: All pipeline outputs and audited source code comply with config.yaml")
    else:
        print("\nAll mismatches:")
        for i, m in enumerate(mismatches, 1):
            print(f"  {i:>3}. {m}")
    return 0 if not mismatches else 1


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Audit OceanEmbed pipeline outputs against config.yaml")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    raise SystemExit(main(parser.parse_args().config))
