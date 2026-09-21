# OceanEmbed

OceanEmbed learns a spatial representation of daily ocean surface observations and reconstructs a depth-resolved temperature field. It harmonizes CMEMS, PO.DAAC, GLORYS, and real ARGO observations onto one grid, trains a CNN encoder/decoder, and validates predictions against real `PLATFORM_NUMBER`-tracked floats.

This repository is an experiment pipeline. The current one-month Arabian Sea run is a smoke test; the validation report flags low-confidence depths and does not treat missing seasons or regions as data.

## Architecture

```mermaid
flowchart TD
    C[config.yaml\npaths, dates, domain, depths, model] --> D[00 Access\nCMEMS / PO.DAAC]
    D --> R[data/raw\nNetCDF files]
    R --> H[01 Harmonize\nregrid, daily mean, shared mask]
    H --> S[surface Zarr\n(time, lat, lon)]
    R --> G[GLORYS target preparation]
    G --> T[target_temperature.zarr\n(time, depth, lat, lon)]
    R --> A[ARGO preparation\nreal platform IDs]
    A --> V[ARGO validation CSV\nfloat-level split]
    S --> N[03 Splits + train-only normalization]
    T --> N
    N --> E[04 Embedding pretraining]
    N --> M[05 Reconstruction training]
    M --> P[06 Test predictions + ARGO validation]
    P --> Q[07 Dashboard\nStreamlit display-only UI]
```

## Setup

Run from the repository root on Windows:

```cmd
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

CMEMS credentials may be supplied through environment variables or the toolbox credential store:

```cmd
set COPERNICUSMARINE_USERNAME=your-cmems-username
set COPERNICUSMARINE_PASSWORD=your-cmems-password
```

PO.DAAC uses Earthdata authentication:

```cmd
python -c "import earthaccess; earthaccess.login(persist=True)"
```

INCOIS is not currently consumed by a download module. If an INCOIS source is added, keep its credentials in environment variables or the provider credential store, never in `config.yaml` or source code.

## Configuration

All paths, dates, domain bounds, target depths, split dates, model settings, validation tolerances, and region bounds live in [config.yaml](config.yaml). Use a copied config to change scope; source code should not be edited for a new domain or period.

Generated `data/`, `processed_data/`, `outputs/`, and `checkpoints/` are ignored by Git.

## Ordered reproduction

Each numbered script accepts one optional argument: `--config PATH`. The default is `config.yaml`.

1. `scripts/00_test_access.py`: one-day SST access check and PNG.
2. `scripts/01_test_harmonization.py`: grid, gaps, ranges, mask, and visual checks.
3. `scripts/02_test_targets.py`: target alignment, ARGO IDs, and GLORYS ceiling metrics.
4. `scripts/03_test_splits.py`: leakage, train-only normalization, tensors, histograms.
5. `scripts/04_test_embedding.py`: pretrained embedding PCA and reconstruction check.
6. `scripts/05_evaluate_training.py`: loss curves and model/climatology comparison.
7. `scripts/06_validation_report.py`: predictions, real-ID ARGO matching, metrics, tests, Taylor plot.
8. `scripts/07_test_dashboard.py`: automated dashboard artifact spot-checks.

The required preparation modules run between these checkpoints:

```cmd
python -m src.download.download_cmems
python -m src.download.download_podaac
python -m src.preprocessing.harmonize
python -m src.data.prepare_target
python -m src.data.prepare_argo_validation
python -m src.models.autoencoder_pretrain
python -m src.train
```

On Windows Command Prompt, use the native batch runner:

```cmd
run_all.bat config.yaml
```

On systems with GNU Make, the equivalent is:

```cmd
make CONFIG=config.yaml all
```

## Outputs

- `data/raw/`: CMEMS, PO.DAAC, and ARGO source NetCDF files.
- `processed_data/oceanembed.zarr`: daily surface cube and shared `ocean_mask`.
- `processed_data/target_temperature.zarr`: 15-depth GLORYS target cube.
- `processed_data/argo_validation.csv`: real platform IDs and float-level split labels.
- `processed_data/predicted_field.zarr`: test-period model predictions.
- `processed_data/argo_validation_summary.json`: overall, seasonal, regional, audit, and significance results.
- `outputs/`: maps, losses, RMSE plots, Taylor diagram, and dashboard checks.
- `checkpoints/`: model checkpoints.

NaN land/missing pixels become neutral standardized zero in model tensors. The shared `ocean_mask` remains available for masked losses and validation. The dashboard only displays saved Phase 6 artifacts; it does not retrain or rematch observations.

## Tests and hardcoding audit

Run synthetic tests without network or full datasets:

```cmd
python -m pytest -q tests
```

Audit `src/` for literals that may indicate hardcoded scope:

```powershell
Get-ChildItem src -Recurse -Filter *.py | Select-String -Pattern '2024-|2025-|65\.0|75\.0|5\.0|15\.0|data[\\/]|depths_m'
```

Expected hits are config lookups and path construction. Any literal date, domain, depth list, or absolute path should be reviewed manually.

## Dashboard

After Phase 6:

```cmd
streamlit run src/dashboard/app.py
```

Open http://localhost:8501.
