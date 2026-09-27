"""Export real OceanEmbed artifacts into compact JSON for the web dashboard.

Reads:
  processed_data/predicted_field.zarr       (time, depth, lat, lon) temperature
  processed_data/argo_validation_summary.json

Writes design/data/{meta,metrics,field,floats}.json

Re-runnable. Run from the repo root:  python design/build_data.py
"""
from __future__ import annotations
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "processed_data"
OUT = Path(__file__).resolve().parent / "data"
OUT.mkdir(parents=True, exist_ok=True)

LON_STRIDE = 2  # downsample lon 241 -> 121 to keep field.json small


def _round(x, n=3):
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else round(f, n)


def _nweighted_rmse(block: dict) -> tuple[float, int]:
    num = 0.0
    n = 0
    for _, v in block.items():
        r, c = v.get("RMSE"), v.get("n_matches")
        if r is None or c is None:
            continue
        num += (r ** 2) * c
        n += c
    return (math.sqrt(num / n) if n else float("nan")), n


def main() -> None:
    summary = json.loads((PROC / "argo_validation_summary.json").read_text())
    ds = xr.open_zarr(PROC / "predicted_field.zarr")

    depths = [float(d) for d in ds.depth.values]
    times = [str(t)[:10] for t in ds.time.values]
    # representative date: mid-window
    rep_date = times[len(times) // 2]

    model_rmse, n_all = _nweighted_rmse(summary["overall"])
    clim_rmse, _ = _nweighted_rmse(summary["climatology"])
    glorys_rmse, _ = _nweighted_rmse(summary["glorys_ceiling"])
    beat = sum(
        summary["overall"][k]["RMSE"] < summary["climatology"][k]["RMSE"]
        for k in summary["overall"]
    )
    r_max = max(summary["overall"][k]["correlation"] for k in summary["overall"])
    audit = summary["platform_audit"]
    tt = summary["paired_ttest"]

    meta = {
        "domain": {"lat": [5.0, 30.0], "lon": [45.0, 105.0],
                   "name": "Arabian Sea + Bay of Bengal"},
        "depths": depths,
        "dates": times,
        "rep_date": rep_date,
        "test_window": "Dec 2023 - Jan 2024",
        "headline": {
            "model_rmse": _round(model_rmse, 2),
            "clim_rmse": _round(clim_rmse, 2),
            "glorys_rmse": _round(glorys_rmse, 2),
            "improvement_pct": _round((1 - model_rmse / clim_rmse) * 100, 1),
            "ceiling_gap": _round(model_rmse - glorys_rmse, 2),
            "depths_beat": beat,
            "depths_total": len(depths),
            "n_obs": int(audit["matched_rows"]),
            "n_floats": int(audit["distinct_platforms"]),
            "r_max": _round(r_max, 2),
            "t_stat": _round(tt["t_statistic"], 1),
            "p_value": tt["p_value"],
        },
    }
    _dump("meta.json", meta)

    write_metrics(summary, depths)
    write_field(ds, depths, rep_date)
    write_floats(summary, depths)
    print("wrote:", *(p.name for p in sorted(OUT.glob("*.json"))))


def _dump(name: str, obj) -> None:
    (OUT / name).write_text(json.dumps(obj, separators=(",", ":")))


def write_metrics(summary: dict, depths: list[float]) -> None:
    def row(block, d):
        v = block.get(f"{d}") or block.get(f"{d:.1f}") or {}
        return {"rmse": _round(v.get("RMSE")), "mae": _round(v.get("MAE")),
                "bias": _round(v.get("bias")), "corr": _round(v.get("correlation")),
                "n": v.get("n_matches")}
    metrics = {
        "depths": depths,
        "model": [row(summary["overall"], d) for d in depths],
        "climatology": [row(summary["climatology"], d) for d in depths],
        "glorys": [row(summary["glorys_ceiling"], d) for d in depths],
        "region": {reg: [row(summary["region"][reg], d) for d in depths]
                   for reg in summary["region"]},
        "season": {s: [row(summary["season"][s], d) for d in depths]
                   for s in summary["season"]},
    }
    _dump("metrics.json", metrics)


def write_field(ds: xr.Dataset, depths: list[float], rep_date: str) -> None:
    da = ds["temperature"].sel(time=np.datetime64(rep_date))
    lats = [float(v) for v in da.lat.values]
    lons = [float(v) for v in da.lon.values[::LON_STRIDE]]
    layers = []
    for d in depths:
        arr = da.sel(depth=d).values[:, ::LON_STRIDE]
        grid = [[None if np.isnan(x) else round(float(x), 2) for x in rowv]
                for rowv in arr]
        layers.append(grid)
    _dump("field.json", {"date": rep_date, "lat": lats, "lon": lons,
                         "depths": depths, "temp": layers})


def write_floats(summary: dict, depths: list[float]) -> None:
    by_float: dict[str, list[dict]] = defaultdict(list)
    for r in summary["matched_records"]:
        by_float[str(r["platform_number"])].append(r)
    floats = []
    for pid, recs in sorted(by_float.items()):
        recs.sort(key=lambda r: (r["depth"], r["time"]))
        lat = float(np.mean([r["lat"] for r in recs]))
        lon = float(np.mean([r["lon"] for r in recs]))
        region = "Arabian Sea" if recs[0].get("arabian_sea") else "Bay of Bengal"
        # mean profile per depth
        prof: dict[float, dict] = {}
        agg: dict[float, dict] = defaultdict(lambda: defaultdict(list))
        for r in recs:
            for k in ("observed", "model", "glorys", "climatology"):
                agg[r["depth"]][k].append(r[k])
        for d in sorted(agg):
            prof[d] = {k: _round(float(np.mean(agg[d][k])), 2)
                       for k in ("observed", "model", "glorys", "climatology")}
        floats.append({
            "id": pid, "lat": round(lat, 2), "lon": round(lon, 2),
            "region": region, "n": len(recs),
            "depths": [d for d in sorted(prof)],
            "observed": [prof[d]["observed"] for d in sorted(prof)],
            "model": [prof[d]["model"] for d in sorted(prof)],
            "glorys": [prof[d]["glorys"] for d in sorted(prof)],
            "climatology": [prof[d]["climatology"] for d in sorted(prof)],
        })
    _dump("floats.json", {"floats": floats})


if __name__ == "__main__":
    main()
