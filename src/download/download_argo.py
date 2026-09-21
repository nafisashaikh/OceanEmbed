"""Download configured ARGO float profiles from Ifremer ERDDAP with monthly chunking."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys
import time
import urllib.parse
from typing import Any

import pandas as pd
import requests
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.download.download_cmems import _get_date_range, _generate_monthly_chunks

logger = logging.getLogger("oceanembed.download_argo")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

ERDDAP_BASE_URL = "https://erddap.ifremer.fr/erddap/tabledap/ArgoFloats.nc"
DEFAULT_ARGO_VARS = [
    "config_mission_number", "cycle_number", "data_mode", "direction",
    "latitude", "longitude", "platform_number", "position_qc", "pres",
    "pres_adjusted", "pres_adjusted_error", "pres_adjusted_qc", "pres_qc",
    "psal", "psal_adjusted", "psal_adjusted_error", "psal_adjusted_qc", "psal_qc",
    "temp", "temp_adjusted", "temp_adjusted_error", "temp_adjusted_qc", "temp_qc",
    "time", "time_qc", "vertical_sampling_scheme"
]


def _is_valid_argo_nc(file_path: Path) -> bool:
    """Check if ARGO NetCDF file exists, has size > 1KB, and contains valid profile rows."""
    if not file_path.is_file() or file_path.stat().st_size < 1024:
        return False
    try:
        with xr.open_dataset(file_path) as ds:
            if "row" in ds.dims and ds.sizes["row"] == 0:
                return False
            if "platform_number" not in ds and "PLATFORM_NUMBER" not in ds:
                return False
            if "temp" not in ds and "TEMP" not in ds and "temp_adjusted" not in ds:
                return False
            return True
    except Exception:
        return False


def _build_erddap_url(
    domain: dict[str, float],
    start_fmt: str,
    end_fmt: str,
    max_depth: float = 1000.0,
) -> str:
    """Construct URL-encoded query string for Ifremer ERDDAP ArgoFloats.nc endpoint."""
    query_str = (
        ",".join(DEFAULT_ARGO_VARS) +
        f"&longitude>={domain['west']}&longitude<={domain['east']}" +
        f"&latitude>={domain['south']}&latitude<={domain['north']}" +
        f"&pres>=0.0&pres<={max_depth}" +
        f"&time>={start_fmt}&time<={end_fmt}" +
        "&latitude!=NaN&longitude!=NaN&.distinct()&.orderBy(%22time,pres%22)"
    )
    return f"{ERDDAP_BASE_URL}?{query_str}"


def download_argo_chunk(
    chunk: dict[str, Any],
    domain: dict[str, float],
    output_dir: Path,
    max_depth: float = 1000.0,
    timeout_s: int = 120,
) -> Path:
    """Download a single monthly chunk of ARGO profiles from ERDDAP."""
    month_label = chunk["month"]
    filename = f"argo_profiles_{month_label.replace('-', '')}.nc"
    target_file = output_dir / filename

    if target_file.exists():
        if _is_valid_argo_nc(target_file):
            size_mb = target_file.stat().st_size / (1024 * 1024)
            logger.info(f"Month {month_label} already exists and is valid ({size_mb:.2f} MB): {filename}. Skipping.")
            return target_file
        else:
            logger.warning(f"Month {month_label} file is corrupted or empty ({filename}). Deleting and re-downloading.")
            target_file.unlink(missing_ok=True)

    url = _build_erddap_url(domain, chunk["start"], chunk["end"], max_depth)
    logger.info(f"Downloading ARGO month {month_label} ({chunk['start'][:10]} to {chunk['end'][:10]}) -> {filename}...")

    start_time = time.time()
    try:
        response = requests.get(url, timeout=timeout_s)
        if response.status_code != 200:
            # Fallback to SSL unverified request if SSL verification failed
            response = requests.get(url, verify=False, timeout=timeout_s)
            response.raise_for_status()

        target_file.write_bytes(response.content)
        elapsed = time.time() - start_time

        if not _is_valid_argo_nc(target_file):
            raise IOError(f"Downloaded file {filename} failed integrity validation (corrupted or 0 profile rows).")

        size_mb = target_file.stat().st_size / (1024 * 1024)
        speed = size_mb / max(elapsed, 0.001)
        with xr.open_dataset(target_file) as ds:
            n_rows = ds.sizes.get("row", 0)
            n_platforms = len(set(ds["platform_number"].values)) if "platform_number" in ds else 0
        logger.info(
            f"SUCCESS: Month {month_label} downloaded ({size_mb:.2f} MB, {n_rows:,} rows across {n_platforms} platforms in {elapsed:.1f}s, {speed:.2f} MB/s)"
        )
        return target_file
    except Exception as exc:
        target_file.unlink(missing_ok=True)
        raise RuntimeError(f"Failed downloading ARGO month {month_label}: {exc}") from exc


def download_all(
    config_path: str | Path = "config.yaml",
    dry_run: bool = False,
) -> None:
    """Download ARGO float profiles for every configured month."""
    config = load_config(config_path)
    domain = config["domain"]
    start_date, end_date = _get_date_range(config)
    chunks = _generate_monthly_chunks(start_date, end_date)
    max_depth = float(config.get("depths", {}).get("maximum", 1000.0))

    output_dir = Path(config["paths"]["raw_data_dir"]) / "argo"
    output_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        print("\n" + "=" * 80)
        print("           OCEANEMBED ARGO DOWNLOAD ESTIMATION REPORT (DRY-RUN)")
        print("=" * 80)
        print(f"Date Range : {start_date} to {end_date}")
        print(f"Total Months: {len(chunks)}")
        print(f"Domain     : Lon [{domain['west']}, {domain['east']}], Lat [{domain['south']}, {domain['north']}], Depth [0, {max_depth}] m")
        print(f"Output Dir : {output_dir}")
        print("-" * 80)
        print(f"{'Month':<9} {'Date Range':<32} {'Filename':<24} {'Status'}")
        print("-" * 80)
        for c in chunks:
            fn = f"argo_profiles_{c['month'].replace('-', '')}.nc"
            p = output_dir / fn
            is_valid = p.exists() and _is_valid_argo_nc(p)
            status_str = f"EXISTS ({(p.stat().st_size / (1024*1024)):.2f} MB)" if is_valid else "PENDING"
            dr_str = f"{c['start'][:10]} to {c['end'][:10]}"
            print(f"{c['month']:<9} {dr_str:<32} {fn:<24} {status_str}")
        print("=" * 80 + "\n")
        return

    succeeded = 0
    skipped = 0
    failed = []

    logger.info(f"Starting ARGO download across {len(chunks)} months ({chunks[0]['month']} to {chunks[-1]['month']})...")
    for chunk in chunks:
        try:
            filename = f"argo_profiles_{chunk['month'].replace('-', '')}.nc"
            target_file = output_dir / filename
            if target_file.exists() and _is_valid_argo_nc(target_file):
                skipped += 1
                logger.info(f"Month {chunk['month']} already valid. Skipping.")
                continue

            download_argo_chunk(chunk, domain, output_dir, max_depth)
            succeeded += 1
        except Exception as exc:
            logger.error(f"Month {chunk['month']} download failed: {exc}")
            failed.append({"month": chunk["month"], "error": str(exc)})

    print("\n" + "=" * 80)
    print("                    ARGO DOWNLOAD SUMMARY")
    print("=" * 80)
    print(f"Months Skipped (Already Valid) : {skipped}")
    print(f"Months Downloaded Successfully : {succeeded}")
    print(f"Months Failed                  : {len(failed)}")
    print("=" * 80)
    if failed:
        for f in failed:
            print(f"  FAILED: {f['month']} -> {f['error']}")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download configured ARGO float profiles.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Report status of ARGO monthly files without downloading")
    args = parser.parse_args()
    download_all(config_path=args.config, dry_run=args.dry_run)
