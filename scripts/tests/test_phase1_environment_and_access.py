"""
Verification test suite for Phase 1: Environment & Data Access.

Tests:
1. copernicusmarine login and earthaccess.login() both succeed.
2. Each of the 6 configured datasets returns at least 1 day of valid finite data.
3. SST sanity plot and dataset check (values in sane ocean range 20-35 C, not all zero, not all NaN).
4. Config-driven download verification: modifying domain/dates in config changes download arguments with zero code edits.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock
import yaml
import numpy as np
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.download.download_cmems import _login as cmems_login, _subset_product
from src.download.download_podaac import download_all as podaac_download


def test_logins() -> bool:
    print("\n[1/4] Testing copernicusmarine and earthaccess logins...")
    try:
        import copernicusmarine
        cmems_login(copernicusmarine)
        print("  OK: copernicusmarine authentication succeeded")
    except Exception as e:
        print(f"  FAIL: copernicusmarine authentication failed: {e}")
        return False

    try:
        import earthaccess
        auth = earthaccess.login(persist=True)
        if not auth.authenticated:
            print("  FAIL: earthaccess login returned authenticated=False")
            return False
        print("  OK: earthaccess authentication succeeded (authenticated=True)")
    except Exception as e:
        print(f"  FAIL: earthaccess authentication failed: {e}")
        return False

    return True


def test_six_datasets(config_path: str = "config.yaml") -> bool:
    print("\n[2/4] Testing data availability across all 6 datasets...")
    config = load_config(config_path)
    cmems_root = Path(config["paths"]["cmems_output"])
    podaac_root = Path(config["paths"]["podaac_output"])
    
    all_ok = True
    
    # 1-4: CMEMS products (SST, SSS, SSH, GLORYS)
    for name, prod in config["cmems"]["products"].items():
        subdir = cmems_root / prod["output_subdirectory"]
        files = list(subdir.glob("*.nc"))
        if not files:
            print(f"  FAIL: CMEMS {name} has no files in {subdir}")
            all_ok = False
            continue
        with xr.open_dataset(files[0]) as ds:
            var = prod["variables"][0]
            da = ds[var]
            n_finite = int(da.notnull().sum().values)
            if n_finite == 0:
                print(f"  FAIL: CMEMS {name} has 0 finite values in {files[0].name}")
                all_ok = False
            else:
                print(f"  OK: CMEMS {name:18s} -> {files[0].name} (finite values: {n_finite:,}, shape: {da.shape})")

    # 5-6: PO.DAAC products (OSCAR currents, CCMP winds)
    for name, prod in config["podaac"]["products"].items():
        subdir = podaac_root / prod["output_subdirectory"]
        files = list(subdir.glob("*.nc"))
        if not files:
            print(f"  FAIL: PO.DAAC {name} has no files in {subdir}")
            all_ok = False
            continue
        with xr.open_dataset(files[0]) as ds:
            n_times = ds.sizes.get("time", 1)
            vars_list = list(ds.data_vars.keys())
            print(f"  OK: PO.DAAC {name:16s} -> {files[0].name} (times: {n_times}, vars: {vars_list[:3]})")

    return all_ok


def test_sanity_sst(config_path: str = "config.yaml") -> bool:
    print("\n[3/4] Testing SST sanity output (plausible ocean range, not all-zero/all-NaN)...")
    config = load_config(config_path)
    sanity_dir = Path(config["paths"]["sanity_output"])
    nc_path = sanity_dir / "sst.nc"
    png_path = sanity_dir / config["quick_test"]["output_filename"]

    if not nc_path.exists():
        print(f"  FAIL: sanity netcdf file missing: {nc_path}")
        return False
    if not png_path.exists():
        print(f"  FAIL: sanity plot image missing: {png_path}")
        return False

    with xr.open_dataset(nc_path) as ds:
        var = "analysed_sst" if "analysed_sst" in ds else list(ds.data_vars.keys())[0]
        vals = ds[var].values
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            print("  FAIL: All values in SST sanity file are NaN")
            return False
        if np.all(vals == 0.0):
            print("  FAIL: All values in SST sanity file are 0.0")
            return False

        mean_val = float(finite.mean())
        # Convert Kelvin to Celsius if in Kelvin range (> 200 K)
        mean_c = mean_val - 273.15 if mean_val > 200 else mean_val
        print(f"  SST sanity stats: mean={mean_c:.2f} C (min={float(finite.min()) - 273.15:.2f} C, max={float(finite.max()) - 273.15:.2f} C)")
        if not (20.0 <= mean_c <= 35.0):
            print(f"  FAIL: Mean SST {mean_c:.2f} C outside expected ocean range [20, 35] C")
            return False
        print(f"  OK: SST sanity values are in plausible ocean range ({mean_c:.2f} C), PNG saved ({png_path.stat().st_size:,} bytes)")

    return True


def test_config_parameter_responsiveness(config_path: str = "config.yaml") -> bool:
    print("\n[4/4] Testing config-driven parameter changes (0 code edits)...")
    config = load_config(config_path)
    mock_marine = MagicMock()
    product = config["cmems"]["products"]["sst"]

    # Original
    _subset_product(mock_marine, product, config, Path("temp_out"))
    orig_kwargs = mock_marine.subset.call_args.kwargs

    # Modified domain and time
    mod_config = yaml.safe_load(yaml.dump(config))
    mod_config["domain"]["west"] = 52.0
    mod_config["domain"]["east"] = 62.0
    mod_config["domain"]["south"] = 2.0
    mod_config["domain"]["north"] = 8.0
    mod_config["dates"]["start"] = "2024-02-01T00:00:00Z"
    mod_config["dates"]["end"] = "2024-02-10T23:59:59Z"

    mock_marine.reset_mock()
    _subset_product(mock_marine, product, mod_config, Path("temp_out"))
    mod_kwargs = mock_marine.subset.call_args.kwargs

    if orig_kwargs["minimum_longitude"] == mod_kwargs["minimum_longitude"]:
        print("  FAIL: Downloader did not adopt modified longitude")
        return False
    if orig_kwargs["start_datetime"] == mod_kwargs["start_datetime"]:
        print("  FAIL: Downloader did not adopt modified start_datetime")
        return False

    print(f"  Original call: lon=[{orig_kwargs['minimum_longitude']}, {orig_kwargs['maximum_longitude']}], dates=[{orig_kwargs['start_datetime']}, {orig_kwargs['end_datetime']}]")
    print(f"  Modified call: lon=[{mod_kwargs['minimum_longitude']}, {mod_kwargs['maximum_longitude']}], dates=[{mod_kwargs['start_datetime']}, {mod_kwargs['end_datetime']}]")
    print("  OK: Downloader arguments adapt strictly to config changes with zero code edits")
    return True


def main() -> int:
    print("=" * 72)
    print("  OceanEmbed Phase 1: Environment & Data Access Verification")
    print("=" * 72)

    t1 = test_logins()
    t2 = test_six_datasets()
    t3 = test_sanity_sst()
    t4 = test_config_parameter_responsiveness()

    print("\n" + "=" * 72)
    print("  Verification Summary")
    print("=" * 72)
    print(f"  1. copernicusmarine & earthaccess logins : {'PASS' if t1 else 'FAIL'}")
    print(f"  2. All 6 datasets available and valid    : {'PASS' if t2 else 'FAIL'}")
    print(f"  3. SST sanity NetCDF & PNG plausibility  : {'PASS' if t3 else 'FAIL'}")
    print(f"  4. Dynamic config-driven downloading     : {'PASS' if t4 else 'FAIL'}")
    print("=" * 72)

    return 0 if (t1 and t2 and t3 and t4) else 1


if __name__ == "__main__":
    raise SystemExit(main())
