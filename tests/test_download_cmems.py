"""Unit tests for monthly chunking, validity checks, and dry-run in download_cmems.py."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.download.download_cmems import (
    _get_date_range,
    _generate_monthly_chunks,
    _is_multidimensional,
    _is_valid_nc_file,
    _get_depth_bounds,
    _chunk_output_filename,
    _estimate_product_size_mb,
    _subset_product_monthly,
    _subset_product_single,
    _subset_product,
    _print_retry_summary,
    download_all,
)


def _make_dummy_nc(
    path: Path,
    var_name: str = "thetao",
    start_date: str = "2022-01-01",
    days: int = 31,
) -> Path:
    """Helper to create a small valid NetCDF file with coordinates and data."""
    times = pd.date_range(start_date, periods=days, freq="D")
    data = np.ones((len(times), 2, 2), dtype=np.float32)
    ds = xr.Dataset(
        data_vars={var_name: (("time", "lat", "lon"), data)},
        coords={
            "time": times,
            "lat": [10.0, 11.0],
            "lon": [70.0, 71.0],
        },
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)
    return path


def test_get_date_range():
    cfg1 = {"dates": {"start": "2022-01-01T00:00:00Z", "end": "2024-01-31T23:59:59Z"}}
    assert _get_date_range(cfg1) == ("2022-01-01T00:00:00Z", "2024-01-31T23:59:59Z")

    cfg2 = {"time": {"start_date": "2022-06-01T00:00:00Z", "end_date": "2022-08-31T23:59:59Z"}}
    assert _get_date_range(cfg2) == ("2022-06-01T00:00:00Z", "2022-08-31T23:59:59Z")

    with pytest.raises(KeyError):
        _get_date_range({})


def test_generate_monthly_chunks():
    chunks = _generate_monthly_chunks("2022-01-01T00:00:00Z", "2022-03-15T12:00:00Z")
    assert len(chunks) == 3
    assert chunks[0]["month"] == "2022-01"
    assert chunks[0]["days"] == 31
    assert chunks[0]["start"] == "2022-01-01T00:00:00Z"
    assert chunks[0]["end"] == "2022-01-31T23:59:59Z"

    # February in non-leap year
    assert chunks[1]["month"] == "2022-02"
    assert chunks[1]["days"] == 28

    # March partial
    assert chunks[2]["month"] == "2022-03"
    assert chunks[2]["end"] == "2022-03-15T12:00:00Z"
    assert chunks[2]["days"] == 15


def test_is_multidimensional():
    assert _is_multidimensional({"use_depth": True})
    assert _is_multidimensional({"use_depth": False, "chunk_monthly": True})
    assert _is_multidimensional({"use_depth": False, "dataset_id": "cmems_mod_glo_phy_my_0.083deg_P1D-m"}, "glorys_temp")
    assert _is_multidimensional({"use_depth": False, "chunk_monthly": True, "dataset_id": "METOFFICE-GLO-SST-L4-REP-OBS-SST"}, "sst")
    assert not _is_multidimensional({"use_depth": False, "chunk_monthly": False, "dataset_id": "METOFFICE-GLO-SST"}, "sst")


def test_subset_product_monthly_surface_variable(tmp_path):
    mock_marine = MagicMock()
    output_dir = tmp_path / "sst_test"
    output_dir.mkdir()

    config = {
        "domain": {"west": 45.0, "east": 105.0, "south": 5.0, "north": 30.0},
        "dates": {"start": "2022-01-01T00:00:00Z", "end": "2022-02-28T23:59:59Z"},
    }
    product = {
        "dataset_id": "METOFFICE-GLO-SST-L4-REP-OBS-SST",
        "output_filename": "sst.nc",
        "output_subdirectory": "sst_test",
        "use_depth": False,
        "chunk_monthly": True,
        "variables": ["analysed_sst"],
    }

    def side_effect(**kwargs):
        out_f = Path(kwargs["output_directory"]) / kwargs["output_filename"]
        days = 31 if "2022-01" in kwargs["output_filename"] else 28
        start_date = "2022-01-01" if "2022-01" in kwargs["output_filename"] else "2022-02-01"
        _make_dummy_nc(out_f, var_name="analysed_sst", start_date=start_date, days=days)

    mock_marine.subset.side_effect = side_effect

    succeeded, skipped, failed = _subset_product_monthly(
        mock_marine, product, config, output_dir, product_name="sst"
    )

    assert len(succeeded) == 2
    assert succeeded[0]["filename"] == "sst_2022-01.nc"
    assert succeeded[1]["filename"] == "sst_2022-02.nc"
    assert len(skipped) == 0
    assert len(failed) == 0


def test_chunk_output_filename():
    assert _chunk_output_filename("glorys.nc", "2022-01") == "glorys_2022-01.nc"
    assert _chunk_output_filename("glorys_2022-01.nc", "2022-01") == "glorys_2022-01.nc"


def test_get_depth_bounds():
    cfg = {"depths": {"minimum": 0.0, "maximum": 500.0, "depths_m": [0.0, 50.0, 500.0]}}
    assert _get_depth_bounds(cfg) == (0.0, 500.0)

    cfg2 = {"depths": {"depths_m": [10.0, 20.0, 300.0]}}
    assert _get_depth_bounds(cfg2) == (10.0, 300.0)


def test_is_valid_nc_file(tmp_path):
    # Non-existent file
    assert not _is_valid_nc_file(tmp_path / "missing.nc")

    # 0-byte file
    stub = tmp_path / "stub.nc"
    stub.write_bytes(b"")
    assert not _is_valid_nc_file(stub)

    # Corrupt garbage file
    corrupt = tmp_path / "corrupt.nc"
    corrupt.write_bytes(b"NOT_A_NETCDF_FILE_HEADER" * 100)
    assert not _is_valid_nc_file(corrupt)

    # Valid NetCDF file
    valid_file = tmp_path / "valid.nc"
    _make_dummy_nc(valid_file, var_name="thetao", start_date="2022-01-01", days=31)
    assert _is_valid_nc_file(valid_file, expected_variables=["thetao"], expected_year=2022, expected_month=1)

    # Missing expected variable
    assert not _is_valid_nc_file(valid_file, expected_variables=["missing_var"])

    # Wrong year/month
    assert not _is_valid_nc_file(valid_file, expected_variables=["thetao"], expected_year=2023, expected_month=5)


def test_subset_product_monthly_skip_and_download(tmp_path):
    mock_marine = MagicMock()
    output_dir = tmp_path / "glorys_test"
    output_dir.mkdir()

    # Pre-create January 2022 as an existing valid file
    jan_file = output_dir / "glorys_temperature_2022-01.nc"
    _make_dummy_nc(jan_file, var_name="thetao", start_date="2022-01-01", days=31)

    # Pre-create February 2022 as a corrupt file (should be deleted and re-downloaded)
    feb_file = output_dir / "glorys_temperature_2022-02.nc"
    feb_file.write_bytes(b"corrupt data" * 200)

    config = {
        "domain": {"west": 45.0, "east": 105.0, "south": 5.0, "north": 30.0},
        "dates": {"start": "2022-01-01T00:00:00Z", "end": "2022-02-28T23:59:59Z"},
        "depths": {"minimum": 0.0, "maximum": 1000.0, "depths_m": [0.0, 100.0, 1000.0]},
    }
    product = {
        "dataset_id": "cmems_mod_glo_phy_my_0.083deg_P1D-m",
        "output_filename": "glorys_temperature.nc",
        "output_subdirectory": "glorys_test",
        "use_depth": True,
        "variables": ["thetao"],
    }

    # When mock_marine.subset is called for February, simulate successful creation of valid February file
    def side_effect(**kwargs):
        out_f = Path(kwargs["output_directory"]) / kwargs["output_filename"]
        _make_dummy_nc(out_f, var_name="thetao", start_date="2022-02-01", days=28)

    mock_marine.subset.side_effect = side_effect

    succeeded, skipped, failed = _subset_product_monthly(
        mock_marine, product, config, output_dir, product_name="glorys_temperature"
    )

    # January should be skipped because it is already valid
    assert len(skipped) == 1
    assert skipped[0]["month"] == "2022-01"

    # February should be re-downloaded
    assert len(succeeded) == 1
    assert succeeded[0]["month"] == "2022-02"
    assert len(failed) == 0

    # Ensure mock_marine.subset was called only once (for February, not January)
    assert mock_marine.subset.call_count == 1
    kwargs = mock_marine.subset.call_args.kwargs
    assert kwargs["start_datetime"] == "2022-02-01T00:00:00Z"
    assert kwargs["output_filename"] == "glorys_temperature_2022-02.nc"


def test_subset_product_monthly_failure_continuation(tmp_path):
    mock_marine = MagicMock()
    output_dir = tmp_path / "glorys_fail_test"
    output_dir.mkdir()

    config = {
        "domain": {"west": 45.0, "east": 105.0, "south": 5.0, "north": 30.0},
        "dates": {"start": "2022-01-01T00:00:00Z", "end": "2022-03-31T23:59:59Z"},
        "depths": {"minimum": 0.0, "maximum": 1000.0, "depths_m": [0.0, 1000.0]},
    }
    product = {
        "dataset_id": "cmems_mod_glo_phy_my_0.083deg_P1D-m",
        "output_filename": "glorys_temperature.nc",
        "output_subdirectory": "glorys_fail_test",
        "use_depth": True,
        "variables": ["thetao"],
    }

    # Month 2 fails, months 1 and 3 succeed
    def side_effect(**kwargs):
        if "2022-02" in kwargs["output_filename"]:
            raise RuntimeError("HTTP 504 Gateway Timeout")
        out_f = Path(kwargs["output_directory"]) / kwargs["output_filename"]
        start_day = "2022-01-01" if "2022-01" in kwargs["output_filename"] else "2022-03-01"
        _make_dummy_nc(out_f, var_name="thetao", start_date=start_day, days=31)

    mock_marine.subset.side_effect = side_effect

    succeeded, skipped, failed = _subset_product_monthly(
        mock_marine, product, config, output_dir, product_name="glorys_temperature"
    )

    # 1 and 3 succeeded, 2 failed
    assert len(succeeded) == 2
    assert [s["month"] for s in succeeded] == ["2022-01", "2022-03"]
    assert len(failed) == 1
    assert failed[0]["month"] == "2022-02"
    assert "Gateway Timeout" in failed[0]["error"]


def test_retry_summary_output(capsys):
    failed = [
        {
            "product": "glorys_temperature",
            "month": "2022-02",
            "start": "2022-02-01T00:00:00Z",
            "end": "2022-02-28T23:59:59Z",
            "error": "Connection reset",
        }
    ]
    _print_retry_summary(failed)
    captured = capsys.readouterr().out
    assert "MONTHS REQUIRING RETRY" in captured
    assert "2022-02" in captured
    assert "Connection reset" in captured


def test_estimate_product_size():
    config = {
        "domain": {"west": 45.0, "east": 105.0, "south": 5.0, "north": 30.0},
        "dates": {"start": "2022-01-01T00:00:00Z", "end": "2022-02-28T23:59:59Z"},
        "depths": {"minimum": 0.0, "maximum": 1000.0, "depths_m": [0.0, 100.0, 1000.0]},
        "preprocessing": {"target_resolution": 0.25},
    }
    product = {
        "dataset_id": "cmems_mod_glo_phy_my_0.083deg_P1D-m",
        "output_filename": "glorys_temperature.nc",
        "use_depth": True,
        "variables": ["thetao"],
    }
    chunks = _generate_monthly_chunks("2022-01-01T00:00:00Z", "2022-02-28T23:59:59Z")
    estimates, method = _estimate_product_size_mb(None, product, config, chunks)
    assert len(estimates) == 2
    assert estimates["2022-01"] > 0
    assert estimates["2022-02"] > 0
    # January has 31 days, Feb has 28 days -> Jan should be larger
    assert estimates["2022-01"] > estimates["2022-02"]
    assert "Theoretical grid" in method


def test_dry_run_flag(tmp_path, capsys):
    config = {
        "paths": {"cmems_output": str(tmp_path)},
        "domain": {"west": 70.0, "east": 71.0, "south": 10.0, "north": 11.0},
        "dates": {"start": "2024-01-01T00:00:00Z", "end": "2024-02-29T23:59:59Z"},
        "depths": {"minimum": 0.0, "maximum": 1000.0, "depths_m": [0.0, 10.0]},
        "preprocessing": {"target_resolution": 0.5},
        "cmems": {
            "products": {
                "glorys_test": {
                    "dataset_id": "glorys_id",
                    "output_filename": "glorys.nc",
                    "output_subdirectory": "glorys",
                    "use_depth": True,
                    "variables": ["thetao"],
                }
            }
        },
    }
    import yaml
    cfg_file = tmp_path / "test_config.yaml"
    cfg_file.write_text(yaml.dump(config))

    with patch("src.download.download_cmems.load_config", return_value=config):
        with patch("copernicusmarine.login"):
            download_all(config_path="dummy_config.yaml", dry_run=True)

    captured = capsys.readouterr().out
    assert "OCEANEMBED CMEMS DOWNLOAD ESTIMATION REPORT (DRY-RUN)" in captured
    assert "glorys_test" in captured
    assert "2024-01" in captured
    assert "2024-02" in captured
    assert "DRY-RUN SUMMARY TOTALS" in captured
