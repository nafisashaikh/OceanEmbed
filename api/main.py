"""OceanEmbed FastAPI backend.

Serves the real validation artifacts (predicted field, ARGO skill metrics) and a
genuine live-inference endpoint that runs the trained U-Net on satellite inputs,
plus the static web dashboard in ``design/``. Everything is derived from the same
checkpoint and processed data the offline pipeline produced — no fabricated data.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import xarray as xr
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from src.config import load_config
from api import inference as inf

ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "design"
PROC = ROOT / "processed_data"
SUMMARY = PROC / "argo_validation_summary.json"

def _sanitize(o):
    """Recursively replace non-finite floats (NaN/Inf) with None."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize(v) for v in o]
    return o


class CleanJSONResponse(JSONResponse):
    """JSON response that renders NaN/Inf as null (Starlette rejects them)."""

    def render(self, content) -> bytes:
        return json.dumps(
            _sanitize(content), ensure_ascii=False, allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")


app = FastAPI(title="OceanEmbed API", version="1.0",
              default_response_class=CleanJSONResponse)

# Allow the dashboard to reach the API even when it is opened from a different
# local origin (e.g. the static http.server on :4599 or a file:// preview).
# The uvicorn server on :4600 is the only process that exposes /api/*, so a
# frontend on another port still resolves live data instead of 404-ing.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _grid2d(a: np.ndarray) -> list:
    """2-D float array -> nested lists, NaN -> None, rounded to 2 dp."""
    a = np.round(np.asarray(a, dtype=float), 2)
    return [[None if v != v else float(v) for v in row] for row in a]


def _n_weighted_rmse(block: dict) -> float:
    num = den = 0.0
    for m in block.values():
        n, r = m.get("n_matches", 0), m.get("RMSE")
        if r is not None and n:
            num += n * r * r
            den += n
    return float((num / den) ** 0.5) if den else float("nan")


@lru_cache(maxsize=1)
def _config() -> dict:
    return load_config(str(ROOT / "config.yaml"))


@lru_cache(maxsize=1)
def _summary() -> dict:
    return json.loads(SUMMARY.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _predicted() -> xr.DataArray:
    return xr.open_zarr(inf.PRED_ZARR)["temperature"].load()


def _depths() -> list[float]:
    return [float(d) for d in _config()["depths"]["depths_m"]]


# --------------------------------------------------------------------------- #
# data endpoints
# --------------------------------------------------------------------------- #
@app.get("/api/health")
def health():
    return {"status": "ok", "checkpoint": inf.CKPT.name,
            "dates": len(inf.available_dates())}


@app.get("/api/meta")
def meta():
    cfg = _config()
    s = _summary()
    depths = _depths()
    overall, clim, ceil = s["overall"], s["climatology"], s["glorys_ceiling"]
    model_rmse = _n_weighted_rmse(overall)
    clim_rmse = _n_weighted_rmse(clim)
    glorys_rmse = _n_weighted_rmse(ceil)
    depths_beat = sum(
        1 for d in depths
        if overall[str(d)].get("RMSE") is not None
        and clim[str(d)].get("RMSE") is not None
        and overall[str(d)]["RMSE"] < clim[str(d)]["RMSE"]
    )
    corrs = [overall[str(d)].get("correlation") for d in depths]
    r_max = max([c for c in corrs if c is not None], default=float("nan"))
    dom = cfg["domain"]
    return {
        "domain": {
            "name": "Arabian Sea + Bay of Bengal",
            "lat_min": dom["south"], "lat_max": dom["north"],
            "lon_min": dom["west"], "lon_max": dom["east"],
            "resolution_deg": cfg.get("preprocessing", {}).get("target_resolution", 0.25),
        },
        "depths": depths,
        "test_window": "Dec 2023 – Jan 2024",
        "predicted_dates": inf.available_dates(),
        "model": inf.model_meta(),
        "headline": {
            "model_rmse": round(model_rmse, 2),
            "clim_rmse": round(clim_rmse, 2),
            "glorys_rmse": round(glorys_rmse, 2),
            "improvement_pct": round(100 * (clim_rmse - model_rmse) / clim_rmse, 1),
            "ceiling_gap": round(model_rmse - glorys_rmse, 2),
            "depths_beat": depths_beat,
            "depths_total": len(depths),
            "n_obs": int(s["paired_ttest"]["n_matches"]),
            "n_floats": int(s["platform_audit"]["distinct_platforms"]),
            "r_max": round(float(r_max), 2),
            "t_stat": round(float(s["paired_ttest"]["t_statistic"]), 1),
            "p_value": float(s["paired_ttest"]["p_value"]),
        },
    }


@app.get("/api/metrics")
def metrics():
    """Per-depth skill for model / climatology / GLORYS + region & season."""
    s = _summary()
    depths = _depths()

    def series(block: dict) -> list:
        out = []
        for d in depths:
            m = block.get(str(d), {})
            out.append({
                "depth": d,
                "rmse": m.get("RMSE"),
                "mae": m.get("MAE"),
                "bias": m.get("bias"),
                "corr": m.get("correlation"),
                "n": m.get("n_matches", 0),
            })
        return out

    regions = {name: series(block) for name, block in s.get("region", {}).items()}
    seasons = {name: series(block) for name, block in s.get("season", {}).items()
               if any(v.get("n_matches") for v in block.values())}
    return {
        "depths": depths,
        "model": series(s["overall"]),
        "climatology": series(s["climatology"]),
        "glorys": series(s["glorys_ceiling"]),
        "regions": regions,
        "seasons": seasons,
        "paired_ttest": s["paired_ttest"],
        "paired_ttest_by_depth": s.get("paired_ttest_by_depth", {}),
    }


@app.get("/api/floats")
def floats():
    """Per-float positions and observed/model/GLORYS/climatology profiles."""
    s = _summary()
    import pandas as pd
    df = pd.DataFrame(s["matched_records"])
    if df.empty:
        return {"floats": []}
    out = []
    for pid, g in df.groupby("platform_number"):
        g = g.sort_values("depth")
        prof = (g.groupby("depth")[["observed", "model", "glorys", "climatology"]]
                  .mean().reset_index().sort_values("depth"))
        err = float(np.sqrt(np.mean((g["model"] - g["observed"]) ** 2)))
        out.append({
            "id": str(pid),
            "lat": round(float(g["lat"].mean()), 3),
            "lon": round(float(g["lon"].mean()), 3),
            "n": int(len(g)),
            "rmse": round(err, 3),
            "depths": [float(x) for x in prof["depth"]],
            "observed": [round(float(x), 3) for x in prof["observed"]],
            "model": [round(float(x), 3) for x in prof["model"]],
            "glorys": [round(float(x), 3) for x in prof["glorys"]],
            "climatology": [round(float(x), 3) for x in prof["climatology"]],
        })
    out.sort(key=lambda f: f["id"])
    return {"floats": out, "count": len(out)}


@app.get("/api/field")
def field(date: str = Query(...), depth_index: int = Query(0, ge=0, le=14)):
    """Canonical stored predicted field: one depth slice for one test date."""
    da = _predicted()
    dates = [str(t)[:10] for t in da.time.values]
    if date not in dates:
        raise HTTPException(404, f"date {date} not in predicted field")
    depths = _depths()
    z = da.isel(time=dates.index(date), depth=depth_index).values
    return {
        "date": date,
        "depth_index": depth_index,
        "depth": depths[depth_index],
        "depths": depths,
        "lat": [round(float(v), 3) for v in da.lat.values],
        "lon": [round(float(v), 3) for v in da.lon.values],
        "z": _grid2d(z),
    }


@app.get("/api/volume")
def volume(date: str = Query(...), stride: int = Query(2, ge=1, le=6)):
    """Full 15-depth reconstructed field for one date, downsampled in lat/lon,
    for the 3D stacked-surface view. `stride` thins the horizontal grid so the
    browser can render all 15 layers smoothly."""
    da = _predicted()
    dates = [str(t)[:10] for t in da.time.values]
    if date not in dates:
        raise HTTPException(404, f"date {date} not in predicted field")
    depths = _depths()
    vol = da.isel(time=dates.index(date)).values  # (depth, lat, lon)
    lat = da.lat.values[::stride]
    lon = da.lon.values[::stride]
    vol = vol[:, ::stride, ::stride]
    return {
        "date": date,
        "depths": depths,
        "lat": [round(float(v), 3) for v in lat],
        "lon": [round(float(v), 3) for v in lon],
        "layers": [_grid2d(vol[k]) for k in range(vol.shape[0])],
    }


@app.get("/api/reconstruct")
def reconstruct(date: str = Query(...), depth_index: int = Query(0, ge=0, le=14)):
    """LIVE U-Net inference for a date. Runs the model now and returns the
    requested depth slice, latency, and agreement vs. the stored field."""
    if date not in inf.available_dates():
        raise HTTPException(404, f"date {date} not reconstructable (test window only)")
    field_arr, depths, latency_ms, emb_norm = inf.reconstruct(date)
    lat, lon = inf.grid_axes()

    # agreement with the canonical stored field (same date), ocean cells only
    da = _predicted()
    d_list = [str(t)[:10] for t in da.time.values]
    agreement = None
    if date in d_list:
        stored = da.isel(time=d_list.index(date)).values
        both = ~np.isnan(field_arr) & ~np.isnan(stored)
        if both.any():
            diff = field_arr[both] - stored[both]
            agreement = {
                "rmse_vs_stored": round(float(np.sqrt(np.mean(diff ** 2))), 4),
                "max_abs_diff": round(float(np.abs(diff).max()), 4),
                "cells": int(both.sum()),
            }

    z = field_arr[depth_index]
    col = np.array([np.nanmean(field_arr[k]) for k in range(field_arr.shape[0])])
    return {
        "date": date,
        "depth_index": depth_index,
        "depth": depths[depth_index],
        "depths": [float(d) for d in depths],
        "latency_ms": round(latency_ms, 1),
        "embedding_norm": round(emb_norm, 3),
        "embedding_dim": inf.model_meta()["embedding_dim"],
        "agreement": agreement,
        "column_mean": [None if v != v else round(float(v), 2) for v in col],
        "lat": [round(float(v), 3) for v in lat],
        "lon": [round(float(v), 3) for v in lon],
        "z": _grid2d(z),
    }


@app.get("/api/dates")
def dates():
    return {"dates": inf.available_dates(), "depths": _depths()}


@app.get("/api/scatter")
def scatter():
    """Real observed-vs-model matchups (every ARGO cast that was paired), for a
    parity diagram. Points and column statistics are computed from the same
    matched_records the validation summary was built from."""
    s = _summary()
    import pandas as pd
    df = pd.DataFrame(s["matched_records"])
    if df.empty:
        return {"observed": [], "model": [], "depth": [], "n": 0}
    obs = df["observed"].to_numpy(dtype=float)
    mod = df["model"].to_numpy(dtype=float)
    dep = df["depth"].to_numpy(dtype=float)
    m = np.isfinite(obs) & np.isfinite(mod)
    obs, mod, dep = obs[m], mod[m], dep[m]
    r = float(np.corrcoef(obs, mod)[0, 1]) if obs.size > 1 else float("nan")
    return {
        "observed": [round(float(x), 2) for x in obs],
        "model": [round(float(x), 2) for x in mod],
        "depth": [round(float(x), 1) for x in dep],
        "n": int(obs.size),
        "rmse": round(float(np.sqrt(np.mean((mod - obs) ** 2))), 3),
        "mae": round(float(np.mean(np.abs(mod - obs))), 3),
        "bias": round(float(np.mean(mod - obs)), 3),
        "r": round(r, 3),
        "r2": round(r * r, 3),
        "tmin": round(float(min(obs.min(), mod.min())), 1),
        "tmax": round(float(max(obs.max(), mod.max())), 1),
    }


_INPUT_LABELS = {
    "sst": ("Sea-surface temperature", "°C"),
    "sss": ("Sea-surface salinity", "PSU"),
    "ssh": ("Sea-surface height", "m"),
    "temperature": ("Surface temperature (0 m)", "°C"),
    "current_u": ("Surface current — eastward", "m/s"),
    "current_v": ("Surface current — northward", "m/s"),
    "wind_u": ("Surface wind — eastward", "m/s"),
    "wind_v": ("Surface wind — northward", "m/s"),
}


@app.get("/api/inputs")
def inputs(date: str = Query(...)):
    """Real domain statistics of the 8 satellite/reanalysis surface fields that
    feed the model, for one date. Values are the actual gridded inputs."""
    surf = inf._surface()
    times = inf._times()
    hits = np.where(times == date)[0]
    if hits.size == 0:
        raise HTTPException(404, f"date {date} not available")
    idx = int(hits[0])
    out = []
    for name in inf.INPUT_VARIABLES:
        a = surf[name].isel(time=idx).values.astype(float)
        finite = a[np.isfinite(a)]
        if finite.size:
            label, unit = _INPUT_LABELS[name]
            out.append({
                "key": name, "label": label, "unit": unit,
                "mean": round(float(finite.mean()), 2),
                "min": round(float(finite.min()), 2),
                "max": round(float(finite.max()), 2),
            })
    return {"date": date, "n_channels": len(out) * (inf.LAG_DAYS + 1), "variables": out}


@app.get("/api/embedding")
def embedding():
    """PCA projection of the pretrained surface embedding — real coordinates
    exported by the training pipeline (one point per day, coloured by month;
    held-out test days flagged). Powers the latent-space page."""
    import pandas as pd
    csv = ROOT / "outputs" / "embedding_pca_coords.csv"
    if not csv.exists():
        raise HTTPException(404, "embedding coordinates not available")
    df = pd.read_csv(csv)
    return {
        "points": [
            {
                "date": str(r.date),
                "month": int(r.month),
                "pc1": round(float(r.PC1), 4),
                "pc2": round(float(r.PC2), 4),
                "held_out": bool(r.is_held_out),
            }
            for r in df.itertuples(index=False)
        ],
        "n": int(len(df)),
        "held_out": int(df["is_held_out"].sum()),
    }


# --------------------------------------------------------------------------- #
# static site (mounted last so /api/* wins)
# --------------------------------------------------------------------------- #
app.mount("/", StaticFiles(directory=str(DESIGN), html=True), name="site")
