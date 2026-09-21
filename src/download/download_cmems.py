"""Download configured Copernicus Marine products with monthly chunking for large datasets."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any

import pandas as pd
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config

logger = logging.getLogger("oceanembed.download_cmems")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def _get_date_range(config: dict[str, Any]) -> tuple[str, str]:
    """Extract (start_date, end_date) string tuple from config."""
    dates_cfg = config.get("dates")
    if isinstance(dates_cfg, dict):
        start = dates_cfg.get("start") or dates_cfg.get("start_date")
        end = dates_cfg.get("end") or dates_cfg.get("end_date")
        if start and end:
            return str(start), str(end)
    time_cfg = config.get("time")
    if isinstance(time_cfg, dict):
        start = time_cfg.get("start_date") or time_cfg.get("start")
        end = time_cfg.get("end_date") or time_cfg.get("end")
        if start and end:
            return str(start), str(end)
    raise KeyError("Config must define start and end dates under 'dates' or 'time'")


def _generate_monthly_chunks(start_str: str, end_str: str) -> list[dict[str, Any]]:
    """Divide a date range into monthly chunk intervals."""
    start_ts = pd.Timestamp(start_str)
    end_ts = pd.Timestamp(end_str)
    if start_ts > end_ts:
        raise ValueError(f"Start date ({start_ts}) is after end date ({end_ts})")

    chunks: list[dict[str, Any]] = []
    current_start = start_ts
    while current_start <= end_ts:
        month_period = pd.Period(current_start, freq="M")
        month_end = month_period.end_time.replace(hour=23, minute=59, second=59)
        if current_start.tzinfo is not None:
            month_end = month_end.tz_localize(current_start.tzinfo)

        chunk_end = min(month_end, end_ts)
        start_fmt = current_start.strftime("%Y-%m-%dT%H:%M:%SZ")
        end_fmt = chunk_end.strftime("%Y-%m-%dT%H:%M:%SZ")
        month_label = current_start.strftime("%Y-%m")
        day_count = (chunk_end.date() - current_start.date()).days + 1

        chunks.append({
            "month": month_label,
            "start": start_fmt,
            "end": end_fmt,
            "start_ts": current_start,
            "end_ts": chunk_end,
            "days": day_count,
        })

        next_month_start = (month_period + 1).start_time.replace(hour=0, minute=0, second=0)
        if current_start.tzinfo is not None:
            next_month_start = next_month_start.tz_localize(current_start.tzinfo)
        current_start = next_month_start

    return chunks


def _is_multidimensional(product: dict[str, Any], product_name: str = "") -> bool:
    """Determine if a product should be downloaded with monthly chunking."""
    if product.get("chunk_monthly") is False:
        return False
    if product.get("chunk_monthly") is True:
        return True
    if product.get("use_depth") is True:
        return True
    dataset_id = str(product.get("dataset_id", "")).lower()
    name_lower = product_name.lower()
    if "glorys" in dataset_id or "glorys" in name_lower:
        return True
    return True


def _is_valid_nc_file(
    file_path: Path,
    expected_variables: list[str] | None = None,
    expected_year: int | None = None,
    expected_month: int | None = None,
) -> bool:
    """Check if file exists on disk and is a valid, complete NetCDF file."""
    if not file_path.is_file():
        return False
    # Avoid 0-byte or tiny corrupt stub files
    if file_path.stat().st_size < 1024:
        return False

    try:
        with xr.open_dataset(file_path) as ds:
            # Check variables
            if expected_variables:
                for var in expected_variables:
                    if var not in ds.variables and var not in ds.data_vars:
                        return False

            # Check time dimension
            if "time" in ds.dims:
                if ds.sizes["time"] == 0:
                    return False
                if expected_year is not None and expected_month is not None:
                    time_vals = pd.to_datetime(ds.time.values)
                    if len(time_vals) == 0:
                        return False
                    first_val = time_vals[0]
                    last_val = time_vals[-1]
                    if (first_val.year, first_val.month) != (expected_year, expected_month) and \
                       (last_val.year, last_val.month) != (expected_year, expected_month):
                        return False

            # Touch payload to ensure HDF5 data streams are readable and not truncated
            var_to_test = None
            if expected_variables:
                for v in expected_variables:
                    if v in ds:
                        var_to_test = v
                        break
            elif ds.data_vars:
                var_to_test = next(iter(ds.data_vars.keys()))

            if var_to_test is not None:
                first_slice = {dim: 0 for dim in ds[var_to_test].dims}
                _ = ds[var_to_test].isel(first_slice).values

            return True
    except Exception:
        return False


def _get_depth_bounds(config: dict[str, Any]) -> tuple[float, float]:
    """Get minimum and maximum depths from config (using minimum/maximum or depths_m)."""
    depths = config.get("depths", {})
    min_depth = depths.get("minimum")
    max_depth = depths.get("maximum")
    depths_m = depths.get("depths_m")
    if min_depth is None and depths_m:
        min_depth = float(min(depths_m))
    if max_depth is None and depths_m:
        max_depth = float(max(depths_m))
    return float(min_depth if min_depth is not None else 0), float(max_depth if max_depth is not None else 0)


def _chunk_output_filename(base_filename: str, month_label: str) -> str:
    """Build monthly chunk filename: e.g. glorys_temperature_2022-01.nc."""
    p = Path(base_filename)
    if month_label in p.stem:
        return base_filename
    suffix = p.suffix or ".nc"
    return f"{p.stem}_{month_label}{suffix}"


def _estimate_product_size_mb(
    marine: Any | None,
    product: dict[str, Any],
    config: dict[str, Any],
    chunks: list[dict[str, Any]],
) -> tuple[dict[str, float], str]:
    """Estimate size in MB for each chunk, returning ({month: size_mb}, estimation_method)."""
    domain = config["domain"]
    lon_span = abs(float(domain["east"]) - float(domain["west"]))
    lat_span = abs(float(domain["north"]) - float(domain["south"]))
    use_depth = bool(product.get("use_depth", False))
    depth_levels = len(config.get("depths", {}).get("depths_m", [1])) if use_depth else 1
    var_count = len(product.get("variables", [1]))

    # Attempt CMEMS API dry-run probe on the first chunk if marine client available
    if marine is not None and chunks:
        sample_chunk = chunks[0]
        try:
            min_depth, max_depth = _get_depth_bounds(config)
            probe_kwargs: dict[str, Any] = {
                "dataset_id": product["dataset_id"],
                "variables": product["variables"],
                "minimum_longitude": domain["west"],
                "maximum_longitude": domain["east"],
                "minimum_latitude": domain["south"],
                "maximum_latitude": domain["north"],
                "start_datetime": sample_chunk["start"],
                "end_datetime": sample_chunk["end"],
                "dry_run": True,
            }
            if use_depth:
                probe_kwargs.update(
                    minimum_depth=min_depth,
                    maximum_depth=max_depth,
                )
            res = marine.subset(**probe_kwargs)
            if hasattr(res, "file_size") and res.file_size is not None and res.file_size > 0:
                mb_per_day = float(res.file_size) / max(sample_chunk["days"], 1)
                estimates = {
                    c["month"]: round(mb_per_day * c["days"], 2) for c in chunks
                }
                return estimates, "CMEMS API dry-run metadata"
        except Exception as exc:
            logger.debug(f"API dry-run probe failed ({exc}); falling back to grid calculation.")

    # Fallback: estimate using spatial resolution & data volume
    resolution = float(config.get("preprocessing", {}).get("target_resolution", 0.083))
    nx = int(lon_span / resolution) + 1
    ny = int(lat_span / resolution) + 1
    # 4 bytes per float32 value + NetCDF container overhead
    bytes_per_day = nx * ny * depth_levels * var_count * 4 * 1.05
    mb_per_day = bytes_per_day / (1024 * 1024)

    estimates = {
        c["month"]: round(mb_per_day * c["days"], 2) for c in chunks
    }
    return estimates, "Theoretical grid volume"


def _dry_run_report(
    marine: Any | None,
    config: dict[str, Any],
    product_filter: str | None = None,
) -> None:
    """Print an upfront estimation report of download sizes across all months without downloading."""
    output_root = Path(config["paths"]["cmems_output"])
    products = config["cmems"]["products"]
    start_date, end_date = _get_date_range(config)
    chunks = _generate_monthly_chunks(start_date, end_date)

    print("\n" + "=" * 80)
    print("           OCEANEMBED CMEMS DOWNLOAD ESTIMATION REPORT (DRY-RUN)")
    print("=" * 80)
    print(f"Overall Configured Date Range : {start_date} to {end_date}")
    print(f"Total Months                  : {len(chunks)} months ({sum(c['days'] for c in chunks)} days)")
    print(f"Domain Bounds                 : Lon [{config['domain']['west']}, {config['domain']['east']}], Lat [{config['domain']['south']}, {config['domain']['north']}]")
    if "depths" in config and "depths_m" in config["depths"]:
        print(f"Depth Range                   : [{config['depths'].get('minimum', 0.0)}, {config['depths'].get('maximum', 1000.0)}] m ({len(config['depths']['depths_m'])} configured levels)")
    print("=" * 80)

    grand_total_est_mb = float(0)
    grand_total_disk_mb = float(0)
    grand_total_needed_mb = float(0)

    for prod_name, product in products.items():
        if product_filter and prod_name != product_filter:
            continue

        is_multi = _is_multidimensional(product, prod_name)
        output_dir = output_root / product["output_subdirectory"]
        variables = product.get("variables", [])

        print(f"\nProduct: {prod_name}")
        print(f"  Dataset ID : {product['dataset_id']}")
        print(f"  Variables  : {', '.join(variables)}")
        print(f"  Type       : {'Multi-dimensional (Monthly Chunks)' if is_multi else '2D / Single Request'}")
        print(f"  Output Dir : {output_dir}")

        if is_multi:
            estimates, method = _estimate_product_size_mb(marine, product, config, chunks)
            print(f"  Size Est.  : Based on {method}")
            print("\n  " + "-" * 76)
            print(f"  {'Month':<9} {'Date Range':<32} {'Days':<6} {'Disk Status':<15} {'Est. Size':<12}")
            print("  " + "-" * 76)

            prod_est_mb = float(0)
            prod_disk_mb = float(0)
            prod_needed_mb = float(0)

            for chunk in chunks:
                m_label = chunk["month"]
                chunk_file = _chunk_output_filename(product["output_filename"], m_label)
                target_path = output_dir / chunk_file
                est_mb = estimates.get(m_label, 0)
                prod_est_mb += est_mb

                start_ts = chunk["start_ts"]
                is_valid = target_path.exists() and _is_valid_nc_file(target_path, variables, start_ts.year, start_ts.month)
                if is_valid:
                    actual_mb = target_path.stat().st_size / (1024 * 1024)
                    status_str = f"EXISTS ({actual_mb:.1f}M)"
                    prod_disk_mb += actual_mb
                else:
                    status_str = "PENDING"
                    prod_needed_mb += est_mb

                date_range_str = f"{chunk['start'][:10]} to {chunk['end'][:10]}"
                print(f"  {m_label:<9} {date_range_str:<32} {chunk['days']:<6} {status_str:<15} {est_mb:>9.2f} MB")

            print("  " + "-" * 76)
            print(f"  Total Estimated Size : {prod_est_mb:,.2f} MB ({prod_est_mb / 1024:.2f} GB)")
            print(f"  Already On Disk (OK) : {prod_disk_mb:,.2f} MB ({prod_disk_mb / 1024:.2f} GB)")
            print(f"  Net Download Needed  : {prod_needed_mb:,.2f} MB ({prod_needed_mb / 1024:.2f} GB)")

            grand_total_est_mb += prod_est_mb
            grand_total_disk_mb += prod_disk_mb
            grand_total_needed_mb += prod_needed_mb
        else:
            # Single request 2D dataset
            single_chunk = [{
                "month": "all",
                "start": start_date,
                "end": end_date,
                "days": sum(c["days"] for c in chunks),
            }]
            estimates, method = _estimate_product_size_mb(marine, product, config, single_chunk)
            est_mb = estimates.get("all", 0)
            target_path = output_dir / product["output_filename"]
            is_valid = target_path.exists() and _is_valid_nc_file(target_path, variables)
            if is_valid:
                actual_mb = target_path.stat().st_size / (1024 * 1024)
                status_str = f"EXISTS ({actual_mb:.1f}M)"
                disk_mb = actual_mb
                needed_mb = float(0)
            else:
                status_str = "PENDING"
                disk_mb = float(0)
                needed_mb = est_mb

            print(f"  Size Est.  : {est_mb:.2f} MB ({method})")
            print(f"  Disk Status: {status_str} ({target_path.name})")

            grand_total_est_mb += est_mb
            grand_total_disk_mb += disk_mb
            grand_total_needed_mb += needed_mb

    print("\n" + "=" * 80)
    print("                            DRY-RUN SUMMARY TOTALS")
    print("=" * 80)
    print(f"  Combined Estimated Dataset Size : {grand_total_est_mb:,.2f} MB ({grand_total_est_mb / 1024:.2f} GB)")
    print(f"  Already Complete on Disk        : {grand_total_disk_mb:,.2f} MB ({grand_total_disk_mb / 1024:.2f} GB)")
    print(f"  Remaining to Download           : {grand_total_needed_mb:,.2f} MB ({grand_total_needed_mb / 1024:.2f} GB)")
    print("=" * 80 + "\n")


def _subset_product_monthly(
    marine: Any,
    product: dict[str, Any],
    config: dict[str, Any],
    output_directory: Path,
    product_name: str = "",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Download a multi-dimensional product in monthly chunks with verification, logging, and error handling."""
    start_date, end_date = _get_date_range(config)
    chunks = _generate_monthly_chunks(start_date, end_date)
    domain = config["domain"]
    min_depth, max_depth = _get_depth_bounds(config)
    output_directory.mkdir(parents=True, exist_ok=True)

    base_filename = product["output_filename"]
    variables = product["variables"]
    use_depth = bool(product.get("use_depth", False))

    succeeded: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []

    logger.info(
        f"Starting monthly download for '{product_name or product.get('dataset_id')}' "
        f"({len(chunks)} months from {chunks[0]['month']} to {chunks[-1]['month']})"
    )

    for chunk in chunks:
        month_label = chunk["month"]
        chunk_filename = _chunk_output_filename(base_filename, month_label)
        target_file = output_directory / chunk_filename
        start_ts = chunk["start_ts"]

        # 1. Check if the output file for that month already exists and is valid
        if target_file.exists():
            if _is_valid_nc_file(target_file, variables, start_ts.year, start_ts.month):
                size_mb = target_file.stat().st_size / (1024 * 1024)
                logger.info(
                    f"[{product_name}] Month {month_label} already exists and is valid ({size_mb:.2f} MB): {chunk_filename}. Skipping."
                )
                skipped.append({
                    "month": month_label,
                    "filename": chunk_filename,
                    "size_mb": size_mb,
                    "path": target_file,
                })
                continue
            else:
                logger.warning(
                    f"[{product_name}] Month {month_label} exists but is corrupted or incomplete ({target_file.name}). Deleting and re-downloading."
                )
                try:
                    target_file.unlink(missing_ok=True)
                except Exception as del_err:
                    logger.warning(f"Could not delete corrupted file {target_file}: {del_err}")

        # 2. Download just that month's data for the full domain and depths from config
        subset_kwargs: dict[str, Any] = {
            "dataset_id": product["dataset_id"],
            "variables": variables,
            "minimum_longitude": domain["west"],
            "maximum_longitude": domain["east"],
            "minimum_latitude": domain["south"],
            "maximum_latitude": domain["north"],
            "start_datetime": chunk["start"],
            "end_datetime": chunk["end"],
            "output_directory": str(output_directory),
            "output_filename": chunk_filename,
            "overwrite": True,
        }
        if use_depth:
            subset_kwargs.update(
                minimum_depth=min_depth,
                maximum_depth=max_depth,
            )

        logger.info(
            f"[{product_name}] Downloading month {month_label} ({chunk['start']} to {chunk['end']}) -> {chunk_filename}..."
        )
        start_time = time.time()
        try:
            marine.subset(**subset_kwargs)
            elapsed = time.time() - start_time

            # Verify downloaded file
            if not target_file.exists():
                raise FileNotFoundError(f"Expected output file not found after download: {target_file}")
            if not _is_valid_nc_file(target_file, variables, start_ts.year, start_ts.month):
                raise IOError(f"Downloaded file {chunk_filename} failed integrity validation.")

            # 3. Log size and time taken
            size_mb = target_file.stat().st_size / (1024 * 1024)
            speed_mb_s = size_mb / max(elapsed, 0.001)
            logger.info(
                f"[{product_name}] SUCCESS: Month {month_label} downloaded ({size_mb:.2f} MB in {elapsed:.1f}s, {speed_mb_s:.2f} MB/s)"
            )
            succeeded.append({
                "month": month_label,
                "filename": chunk_filename,
                "size_mb": size_mb,
                "elapsed_s": elapsed,
                "path": target_file,
            })
        except Exception as exc:
            # 4. On failure, log which month failed and continue to the next one
            elapsed = time.time() - start_time
            logger.error(
                f"[{product_name}] FAILED: Month {month_label} ({chunk['start']} to {chunk['end']}): {exc}"
            )
            failed.append({
                "product": product_name or product.get("dataset_id", "unknown"),
                "month": month_label,
                "start": chunk["start"],
                "end": chunk["end"],
                "error": str(exc),
                "elapsed_s": elapsed,
            })
            continue

    return succeeded, skipped, failed


def _subset_product_single(
    marine: Any,
    product: dict[str, Any],
    config: dict[str, Any],
    output_directory: Path,
    product_name: str = "",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Download a 2D or single-request product with integrity check and logging."""
    domain = config["domain"]
    start_date, end_date = _get_date_range(config)
    output_directory.mkdir(parents=True, exist_ok=True)

    target_file = output_directory / product["output_filename"]
    variables = product["variables"]

    succeeded: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []

    if target_file.exists():
        if _is_valid_nc_file(target_file, variables):
            size_mb = target_file.stat().st_size / (1024 * 1024)
            logger.info(
                f"[{product_name}] File already exists and is valid ({size_mb:.2f} MB): {target_file.name}. Skipping."
            )
            skipped.append({
                "month": "all",
                "filename": product["output_filename"],
                "size_mb": size_mb,
                "path": target_file,
            })
            return succeeded, skipped, failed
        else:
            logger.warning(
                f"[{product_name}] Existing file is corrupted or incomplete ({target_file.name}). Deleting and re-downloading."
            )
            try:
                target_file.unlink(missing_ok=True)
            except Exception as del_err:
                logger.warning(f"Could not delete corrupted file {target_file}: {del_err}")

    subset_kwargs: dict[str, Any] = {
        "dataset_id": product["dataset_id"],
        "variables": variables,
        "minimum_longitude": domain["west"],
        "maximum_longitude": domain["east"],
        "minimum_latitude": domain["south"],
        "maximum_latitude": domain["north"],
        "start_datetime": start_date,
        "end_datetime": end_date,
        "output_directory": str(output_directory),
        "output_filename": product["output_filename"],
        "overwrite": True,
    }
    if product.get("use_depth", False):
        min_depth, max_depth = _get_depth_bounds(config)
        subset_kwargs.update(
            minimum_depth=min_depth,
            maximum_depth=max_depth,
        )

    logger.info(
        f"[{product_name}] Downloading {start_date} to {end_date} -> {product['output_filename']}..."
    )
    start_time = time.time()
    try:
        marine.subset(**subset_kwargs)
        elapsed = time.time() - start_time
        if target_file.exists() and _is_valid_nc_file(target_file, variables):
            size_mb = target_file.stat().st_size / (1024 * 1024)
            speed_mb_s = size_mb / max(elapsed, 0.001)
            logger.info(
                f"[{product_name}] SUCCESS: Downloaded {target_file.name} ({size_mb:.2f} MB in {elapsed:.1f}s, {speed_mb_s:.2f} MB/s)"
            )
            succeeded.append({
                "month": "all",
                "filename": product["output_filename"],
                "size_mb": size_mb,
                "elapsed_s": elapsed,
                "path": target_file,
            })
        else:
            raise IOError(f"Downloaded file {target_file.name} failed integrity check or is missing.")
    except Exception as exc:
        elapsed = time.time() - start_time
        logger.error(f"[{product_name}] FAILED: {exc}")
        failed.append({
            "product": product_name or product.get("dataset_id", "unknown"),
            "month": "all",
            "start": start_date,
            "end": end_date,
            "error": str(exc),
            "elapsed_s": elapsed,
        })

    return succeeded, skipped, failed


def _subset_product(
    marine: Any,
    product: dict[str, Any],
    config: dict[str, Any],
    output_directory: Path,
    product_name: str = "",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Download product data using monthly chunks if multi-dimensional, or single request otherwise."""
    if _is_multidimensional(product, product_name):
        return _subset_product_monthly(marine, product, config, output_directory, product_name)
    return _subset_product_single(marine, product, config, output_directory, product_name)


def _print_retry_summary(failed_records: list[dict[str, Any]]) -> None:
    """Print an alert table of any months that failed and need to be retried."""
    if not failed_records:
        return

    print("\n" + "=" * 80)
    print("          CMEMS DOWNLOAD SUMMARY: MONTHS REQUIRING RETRY")
    print("=" * 80)
    print(f"Total Failed Chunks: {len(failed_records)}")
    print("-" * 80)
    print(f"{'Product':<20} {'Month':<9} {'Date Range':<32} {'Error Details'}")
    print("-" * 80)
    for rec in failed_records:
        date_range_str = f"{rec['start'][:10]} to {rec['end'][:10]}"
        print(f"{rec['product']:<20} {rec['month']:<9} {date_range_str:<32} {rec['error']}")
    print("=" * 80 + "\n")


def _login(marine: Any) -> None:
    """Authenticate with Copernicus Marine Service."""
    username = os.environ.get("COPERNICUSMARINE_USERNAME")
    password = os.environ.get("COPERNICUSMARINE_PASSWORD")
    if username and password:
        marine.login(username=username, password=password)
    else:
        marine.login()


def download_all(
    config_path: str | Path = "config.yaml",
    dry_run: bool = False,
    product_name: str | None = None,
) -> None:
    """Authenticate and download every configured CMEMS product."""
    import copernicusmarine

    config = load_config(config_path)

    if dry_run:
        # Perform estimation report without downloading anything
        try:
            _login(copernicusmarine)
            _dry_run_report(copernicusmarine, config, product_filter=product_name)
        except Exception as exc:
            logger.info(f"Logging in for dry-run failed or skipped ({exc}); computing dry-run estimates offline...")
            _dry_run_report(None, config, product_filter=product_name)
        return

    _login(copernicusmarine)
    output_root = Path(config["paths"]["cmems_output"])

    all_succeeded: list[dict[str, Any]] = []
    all_skipped: list[dict[str, Any]] = []
    all_failed: list[dict[str, Any]] = []

    for name, product in config["cmems"]["products"].items():
        if product_name and name != product_name:
            continue
        output_directory = output_root / product["output_subdirectory"]
        succ, skip, fail = _subset_product(copernicusmarine, product, config, output_directory, product_name=name)
        all_succeeded.extend(succ)
        all_skipped.extend(skip)
        all_failed.extend(fail)

    # Print final operational summary
    print("\n" + "=" * 80)
    print("                     CMEMS DOWNLOAD SESSION SUMMARY")
    print("=" * 80)
    print(f"Chunks Skipped (Already Valid on Disk) : {len(all_skipped)}")
    total_downloaded_mb = sum(s["size_mb"] for s in all_succeeded)
    total_downloaded_time = sum(s.get("elapsed_s", 0) for s in all_succeeded)
    print(f"Chunks Downloaded Successfully         : {len(all_succeeded)} ({total_downloaded_mb:.2f} MB in {total_downloaded_time:.1f}s)")
    print(f"Chunks Failed                          : {len(all_failed)}")
    print("=" * 80)

    if all_failed:
        _print_retry_summary(all_failed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download configured Copernicus Marine products.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Estimate total download size across all months without downloading anything",
    )
    parser.add_argument(
        "--product",
        default=None,
        help="Optionally restrict download or estimate to one product name",
    )
    args = parser.parse_args()
    download_all(config_path=args.config, dry_run=args.dry_run, product_name=args.product)
