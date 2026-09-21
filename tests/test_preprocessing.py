from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

from src.preprocessing.harmonize import (
    _regrid_daily,
    _month_key_from_path,
    _group_files_by_month,
    _standardize_dataarray,
)


def test_regrid_daily_bilinear_center_value():
    source = xr.DataArray(
        [[[1.0, 2.0], [3.0, 4.0]]],
        dims=("time", "lat", "lon"),
        coords={"time": pd.date_range("2024-01-01", periods=1), "lat": [0.0, 1.0], "lon": [0.0, 1.0]},
    )
    result = _regrid_daily(source, np.array([0.0, 0.5, 1.0]), np.array([0.0, 0.5, 1.0]))
    assert result.shape == (1, 3, 3)
    assert np.isclose(result.sel(lat=0.5, lon=0.5).item(), 2.5)


def test_month_key_extraction_monthly_and_daily():
    monthly_path = Path("cmems/ssh/ssh_2022-03.nc")
    daily_path = Path("podaac/ccmp_winds/CCMP_Wind_Analysis_20220315_V03.1_L4.nc")
    assert _month_key_from_path(monthly_path) == "2022-03"
    assert _month_key_from_path(daily_path) == "2022-03"

    grouped = _group_files_by_month([monthly_path, daily_path, Path("data/raw/cmems/sst/sst_2022-04.nc")])
    assert list(grouped.keys()) == ["2022-03", "2022-04"]
    assert len(grouped["2022-03"]) == 2
    assert len(grouped["2022-04"]) == 1


def test_standardize_dataarray_depth_minimum_lazy():
    # 4D data with depth
    da = xr.DataArray(
        np.zeros((2, 3, 2, 2)),
        dims=("time", "depth", "latitude", "longitude"),
        coords={
            "time": pd.date_range("2022-01-01", periods=2),
            "depth": [5.0, 15.0, 25.0],
            "latitude": [10.0, 11.0],
            "longitude": [60.0, 61.0],
        },
    )
    ds = xr.Dataset({"thetao": da})
    input_cfg = {"variable": "thetao", "units": "celsius", "depth": "minimum"}
    standardized = _standardize_dataarray(ds, input_cfg)
    assert "depth" not in standardized.dims
    assert standardized.dims == ("time", "lat", "lon")
    assert standardized.shape == (2, 2, 2)
