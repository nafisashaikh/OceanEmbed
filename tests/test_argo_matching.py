from types import SimpleNamespace

import numpy as np
import pandas as pd
import xarray as xr

from src.config import load_config
from src.validation.argo_validation import _match_record


def test_argo_space_time_match_uses_nearest_values():
    config = load_config()
    config["depths"]["depths_m"] = [10.0]
    predicted = xr.DataArray(np.array([[[[20.0]]]]), dims=("time", "depth", "lat", "lon"), coords={"time": [pd.Timestamp("2024-01-25")], "depth": [10.0], "lat": [10.0], "lon": [70.0]})
    glorys = predicted + 1.0
    climate = xr.DataArray(np.array([[[[18.0]]]]), dims=("month", "depth", "lat", "lon"), coords={"month": [1], "depth": [10.0], "lat": [10.0], "lon": [70.0]})
    row = SimpleNamespace(platform_number="FLOAT-1", time=pd.Timestamp("2024-01-25T12:00:00Z"), lat=10.1, lon=70.1, depth=10.5, temperature=19.0)
    matched = _match_record(config, predicted, glorys, climate, row)
    assert matched is not None
    assert matched["platform_number"] == "FLOAT-1"
    assert matched["model"] == 20.0
    assert matched["glorys"] == 21.0
    assert matched["climatology"] == 18.0


def test_argo_space_time_match_rejects_far_time():
    config = load_config()
    config["depths"]["depths_m"] = [10.0]
    field = xr.DataArray(np.ones((1, 1, 1, 1)), dims=("time", "depth", "lat", "lon"), coords={"time": [pd.Timestamp("2024-01-25")], "depth": [10.0], "lat": [10.0], "lon": [70.0]})
    row = SimpleNamespace(platform_number="FLOAT-1", time=pd.Timestamp("2024-02-01", tz="UTC"), lat=10.0, lon=70.0, depth=10.0, temperature=1.0)
    climate = xr.DataArray(np.ones((1, 1, 1, 1)), dims=("month", "depth", "lat", "lon"), coords={"month": [1], "depth": [10.0], "lat": [10.0], "lon": [70.0]})
    assert _match_record(config, field, field, climate, row) is None


def test_build_erddap_url():
    from src.download.download_argo import _build_erddap_url
    domain = {"west": 45.0, "east": 105.0, "south": 5.0, "north": 30.0}
    url = _build_erddap_url(domain, "2022-01-01T00:00:00Z", "2022-01-31T23:59:59Z", 1000.0)
    assert "erddap.ifremer.fr" in url
    assert "longitude%3E=45.0" in url or "longitude>=45.0" in url
    assert "pres%3E=0.0" in url or "pres>=0.0" in url
    assert "2022-01-01" in url


def test_is_valid_argo_nc(tmp_path):
    from src.download.download_argo import _is_valid_argo_nc
    non_existent = tmp_path / "missing.nc"
    assert not _is_valid_argo_nc(non_existent)

    stub = tmp_path / "stub.nc"
    stub.write_bytes(b"short stub")
    assert not _is_valid_argo_nc(stub)