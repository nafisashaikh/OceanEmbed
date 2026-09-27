# OceanEmbed — Complete Project Report

**Smart India Hackathon 2026 · Depth-Resolved Ocean Temperature from Surface Observations**
Domain: Arabian Sea + Bay of Bengal (5°N–30°N, 45°E–105°E) · Report date: 2026-09-26

---

## 1. Executive Summary

OceanEmbed is an end-to-end machine-learning pipeline that reconstructs the **full 3D subsurface ocean temperature field (0–1000 m, 15 depth levels)** using only satellite-observable **surface** data. Direct subsurface measurement is expensive and sparse (ARGO floats); OceanEmbed makes it possible to estimate a continuous, depth-resolved temperature field across a large domain from data that is already collected daily.

**Headline result:** against **real in-situ ARGO floats**, the model achieves **0.87 °C RMSE** — a **35.9% improvement** over the climatology baseline (1.36 °C) — and it **beats the baseline at 14 of 15 depth levels** (all but the deepest, 1000 m). The improvement is statistically significant (paired t-test, p ≈ 0) and sits within **0.14 °C** of the practical upper bound set by the GLORYS reanalysis (0.73 °C).

---

## 2. The Problem and Our Solution

### The problem
Monitoring the subsurface ocean is critical for climate modelling, cyclone-intensity forecasting, and marine-ecosystem preservation. But depth-resolved measurements rely on sparse, costly in-situ sensors. **How do we estimate continuous, high-resolution, depth-resolved temperature fields using only readily available satellite surface observations?**

### Our solution
A **CNN encoder–decoder (U-Net style)** that maps 2D surface fields to a 3D temperature volume:

- **Inputs (surface, satellite-derived):** SST, SSH, SSS, surface winds (U/V), ocean currents (U/V).
- **Harmonization pipeline:** auto-downloads from CMEMS and PO.DAAC, aligns temporal frequencies to daily means, regrids to a uniform 0.25° grid, applies one shared land/ocean mask.
- **Model:** `OceanEmbedReconstructionModel` — a `SurfaceAutoencoder` (pretrained embeddings) plus a mirrored decoder that expands the surface representation into 15 depth channels.
- **Validation:** independent, leak-free comparison against real ARGO floats tracked by `PLATFORM_NUMBER`.

---

## 3. Honest Positioning — Is This Novel?

**The fundamental approach is not new.** Reconstructing subsurface temperature from satellite surface data with CNNs is an established research area (multiple peer-reviewed papers use the same SST+SSH+SSS → subsurface-T mapping). Being upfront about this makes the team credible if a judge cites the literature.

**What is genuinely differentiating (our defensible angle):**

1. **Region — the strongest "why India" hook.** Most published work targets the Pacific, South China Sea, or the global ocean. The **Arabian Sea + Bay of Bengal** (monsoon- and cyclone-critical waters) is far less covered.
2. **Rigorous leak-free validation** against real ARGO floats, segregated by `PLATFORM_NUMBER`, with a strict temporal split (train ≤ Nov 2023; val/test Dec 2023–Jan 2024). Many quick projects skip this.
3. **End-to-end engineering, not a notebook.** Automated CMEMS/PO.DAAC harmonization → embedding pretraining → reconstruction → validation → an interactive decision-ready dashboard.

**Pitch line:** *"We adapted a proven approach to the Indian Ocean with rigorous, leak-free validation and a deployable pipeline"* — honest and still strong.

> Note: a public repo `jyoti-codessss/OceanEmbed-MVP` (SIH 2026, same name/problem) exists on GitHub. Confirm ownership before presenting so any overlap is expected, not a surprise.

---

## 4. Main Results

Validated against **real in-situ ARGO floats** (source: `processed_data/argo_validation_summary.json`).

| Metric | Value | Meaning |
|---|---|---|
| **Model RMSE** | **0.87 °C** | Error vs real ARGO observations |
| Climatology baseline | 1.36 °C | The "no-ML" comparison — model cuts error ~36% |
| GLORYS ceiling | 0.73 °C | Best achievable (the reanalysis it learned from) — model within 0.14 °C |
| Depths beating baseline | **14 / 15** | Wins at every level except the deepest (1000 m) |
| Matched observations | 7,643 | Real ARGO measurements |
| Distinct floats | 13 | All carry `PLATFORM_NUMBER` (leak-free) |
| Paired t-test | t = −51.4, **p ≈ 0** | Improvement is statistically significant |

**Accuracy by depth (highlights):**
- **Surface–30 m:** excellent — RMSE 0.23–0.39 °C, correlation up to **r = 0.98**.
- **Thermocline 75–100 m:** the hardest zone — RMSE ~1.3 °C, r ~0.78. *Physically expected* — steepest, most variable gradients. Still beats climatology (1.30 vs 1.83 at 75 m).
- **Deep 300–700 m:** strong again — RMSE 0.43–0.74 °C, r ~0.92. **The one exception:** at 1000 m the baseline edges ahead (model 0.92 °C vs climatology 0.87 °C) — the single level where climatology wins.

**Three claims to make:** (1) beats baseline at 14 of 15 depths (all but 1000 m), statistically significant; (2) within 0.14 °C of the reanalysis ceiling; (3) honest about the thermocline being hardest — which builds trust.

---

## 5. System Architecture and Data Flow

**Modular pipeline (numbered stages):**
1. **Data ingestion** — Copernicus Marine (CMEMS) + NASA PO.DAAC APIs. *Credentials live only in environment variables, never in config or source.*
2. **Preprocessing / harmonization** — NetCDF → regrid to 0.25°, daily mean, shared land/ocean mask → chunked Zarr stores (`oceanembed.zarr`, `target_temperature.zarr`).
3. **Modeling engine** — PyTorch U-Net: `SurfaceAutoencoder` pretraining, then full reconstruction.
4. **Validation + dashboard** — ARGO matching + a Streamlit interactive dashboard.

**Data sources:** SST (METOFFICE-GLO-SST-L4), SSH (DUACS L4), SSS (multi L4), Winds (CCMP V3.1), Currents (OSCAR V2.0); training target GLORYS12V1; validation truth = ARGO floats.

**Depth levels (15):** 0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000 m.

**Evaluation:** RMSE / MAE / bias / correlation per depth; baseline = climatology; matching tolerances depth 10 m, spatial 0.5°, time 1 day; Taylor diagrams for variance + correlation.

---

## 6. Time Complexity (a talking point judges like)

The whole pipeline is **linear** — no quadratic or exponential blowup.

| Stage | Complexity |
|---|---|
| Harmonize / regrid | `O(T · C · H · W)` |
| U-Net forward (one day) | `O(H · W · K)` |
| Training | `O(E · T · H · W · K)` |
| Inference (full field) | `O(T · H · W · D · K)` |
| ARGO matching | `O(N)` (bounded tolerance window) |

Where `T`=days, `H×W`≈100×240 grid cells, `D`=15 depths, `C`≈7 channels, `E`=epochs, `K`=fixed conv factor, `N`=ARGO profiles.

**Why it's good:** you must touch every grid cell at least once to reconstruct the field, so linear is optimal. A CNN shares weights across space (cost ∝ pixels, not pixels²) — a fully-connected or attention model over the grid would be `O((H·W)²)` and blow up. **Doubling the domain or time range only doubles the compute.** Caveat to state honestly: linear in the good variables, but the constant factor `K` (network size) still governs real speed; training carries the one-time `E` multiplier.

---

## 7. The Dashboard — Pages and Content

The Streamlit dashboard (`src/dashboard/app.py`) is the demo surface. Navigation is ordered **impact-first**:

| # | Page | What it shows |
|---|---|---|
| 1 | **Overview** | Guided landing: "The problem / Our approach / The result" cards + a 30-second orientation and a guide to every page. The first thing judges see. |
| 2 | **Skill summary** | The proof. Headline metrics, RMSE-by-depth chart, Taylor diagram, paired t-test, GLORYS-ceiling context. **CSV download** of RMSE-by-depth. |
| 3 | **Float explorer** | Pick a real ARGO float → observed vs model vs GLORYS profile, and a time-series at any depth the float actually sampled. **CSV download** of the float's matched records. |
| 4 | **Domain map** | Top-down temperature map. Three views: single depth, compare two depths (shared colour scale), and **animate through depths (play button)**. Marine-heatwave detection + **CSV download** of the map slice. |
| 5 | **3D ocean** | Interactive stacked 3D volume of selected depth layers, with **per-point hover showing temperature, lat/lon, and depth**. |
| 6 | **Depth transect** | Vertical temperature slice — warm surface over cool deep water. |
| 7 | **Ocean state** | Operational diagnostics. |
| 8 | **Data pipeline** | The system + data-flow story (regrid → daily mean → embedding → decode to 15 depths). |
| 9 | **Embedding** | The pretrained surface embedding. |

**Design language:** minimal, flat, technical — an "Ocean Intelligence" light palette (bg #F4F8FC, navy sidebar, accent #2563EB). Plus Jakarta Sans headings + Inter body. Turbo colour scale for temperature fields.

---

## 8. Changes Made So Far (this work cycle)

**New features added**
- **3D ocean hover** — every point now reports temperature, lat/lon, and depth. (Fixed a Plotly `go.Surface` limitation: it does not substitute `%{text}` in a hovertemplate, so the label is baked into a per-vertex text array with `hoverinfo="text"`.)
- **Domain map — animation** — a Play/Pause control that descends through all 15 depths on one shared colour scale, plus a depth slider.
- **Domain map — compare two depths** side by side on a shared scale.
- **Float explorer — depth fix** — the time-series depth selector now lists only depths the selected float actually sampled (removes the "no data for this float/depth" warning) and defaults to the shallowest available.
- **CSV downloads** on three pages — map slice, RMSE-by-depth, and a float's matched records. (Server-side PNG export is unavailable because `kaleido` isn't installed; Plotly charts still have the client-side camera button.)
- **Overview landing page** — a guided first page framing problem → approach → result.
- **Impact-first navigation reorder** — Overview → Skill summary → Float explorer first, so judges see proof early.

**"De-AI" redesign** (make it look built by a lab, not generated)
- Removed fake control-room theatrics — no "STATUS: NOMINAL · LIVE" bar, no pulsing live dot, no uppercase "SUBSURFACE OPS".
- Removed emoji-as-icons across headers, verdict badges, metric labels, download buttons, expanders, and info boxes.
- Flattened glossy styling — dropped gradient/glow badges, reduced 24px radii to 10px, softened title weights (800→700), flat white cards with a thin accent bar.
- Sentence-case headings and de-marketed copy ("The 30-second story" → "In brief").
- Consistent flat top-legend across profile, time-series, and transect charts; removed the redundant legend on the source-comparison bar chart.

**Verification:** all 9 pages run headless with **0 exceptions** (Streamlit `AppTest`).

**Known cosmetic notes:** the "Deploy" label seen in one screenshot is a browser-extension overlay, not the app. A `st.components.v1.html` deprecation warning is cosmetic only.

---

## 9. Presentation Guide

### Suggested slide flow (8–10 slides)
1. **Title** — OceanEmbed · depth-resolved ocean temperature from surface satellite data · team + SIH 2026.
2. **The problem** — subsurface data is expensive and sparse (ARGO); it matters for cyclones, climate, fisheries. One strong visual of sparse floats vs a dense surface map.
3. **The idea** — surface is observed daily by satellites; learn the surface → depth mapping. One diagram: 6 surface fields in → 15-depth volume out.
4. **How it works** — the pipeline (harmonize → embed → decode) and the U-Net. Keep it one clean flow diagram.
5. **The result (the money slide)** — 0.87 °C RMSE vs real ARGO, 36% better than baseline, 14/15 depths, p ≈ 0, within 0.14 °C of the ceiling. Big numbers, few words.
6. **Proof of rigor** — leak-free split by `PLATFORM_NUMBER`, validated on floats never seen in training. Show the RMSE-by-depth chart + Taylor diagram.
7. **Live demo** — the dashboard (see flow below).
8. **Why India / why the Indian Ocean** — monsoon + cyclone relevance; most literature ignores this basin.
9. **Honesty + limitations** — thermocline is hardest (say why), bounded by GLORYS as ground truth, scope is two seas.
10. **Roadmap** — real-time inference, more variables (salinity), wider domain, operational alerts.

### Live demo flow (2–3 minutes)
Overview (framing) → **Skill summary** (the proof, download the RMSE CSV to show it's real) → **Float explorer** (pick a float, show observed vs model tracking closely) → **Domain map** (hit Play, descend through depths) → **3D ocean** (rotate, hover to show a real temperature at depth). End on the 3D view — it's the "wow".

### Talking points that land
- *"We don't just beat the baseline on average — we beat it at 14 of 15 depths, all but the very deepest, and a paired t-test says p ≈ 0. This is real skill, not noise."*
- *"We're within 0.14 °C of the reanalysis we learned from — near the ceiling of what's achievable from surface data."*
- *"Validation is leak-free: a float seen in training is never used in testing. The 0.87 °C is against real ocean measurements."*
- *"It's linear-time and CNN-based, so it scales to the whole Indian Ocean without quadratic cost."*

### Likely judge questions (and answers)
- **"Is this novel?"** — The approach exists in literature; our contribution is the Indian-Ocean adaptation, leak-free ARGO validation, and a deployable end-to-end pipeline. (Honest, prepared.)
- **"Why is the thermocline worse?"** — Steepest, most variable gradients; small vertical shifts cause large errors. We still beat climatology there.
- **"What's your ground-truth limit?"** — GLORYS reanalysis; we're within 0.14 °C of it. Beyond that needs more in-situ data.
- **"Real-time?"** — Inference is a single linear forward pass, so yes, feasible; the current demo runs on a fixed test window (Dec 2023–Jan 2024).
- **"Data security?"** — API credentials are only in environment variables, never in source or config.

---

## 10. Limitations and Future Work

**Limitations (state them — it builds trust):**
- Ground truth is capped by the GLORYS reanalysis quality.
- Thermocline (75–100 m) is the least accurate zone.
- Scope is currently two seas and a fixed test window.
- Temperature only (not yet salinity or currents at depth).

**Future work:**
- Real-time / operational inference and alerting (e.g., marine-heatwave and cyclone-intensity signals).
- Extend to salinity and the full Indian Ocean.
- Add physical-consistency constraints (physically-guided loss).
- Longer validation period and more ARGO floats.

---

## 11. One-Paragraph Abstract (for the submission form)

> OceanEmbed reconstructs the full depth-resolved ocean temperature field (0–1000 m, 15 levels) over the Arabian Sea and Bay of Bengal using only satellite-observable surface data (SST, SSH, SSS, winds, currents). A CNN encoder–decoder, trained on GLORYS reanalysis with train-only normalization and a strict temporal split, is validated leak-free against real in-situ ARGO floats. It achieves 0.87 °C RMSE — 35.9% better than the climatology baseline, beating it at 14 of 15 depths (paired t-test p ≈ 0) — within 0.14 °C of the reanalysis ceiling. An interactive dashboard turns the reconstructed field into a decision-ready tool with 3D visualization, per-float validation, depth animation, and downloadable data.





