# Graph Report - SIH PROJECT  (2026-09-20)

## Corpus Check
- 41 files · ~19,413 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 5 file(s) not represented in the graph (top: (none) 3, .bat 1, .nc 1)

## Summary
- 383 nodes · 912 edges · 26 communities (14 shown, 12 thin omitted)
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 5 edges (avg confidence: 0.89)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `7d01f890`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- pathlib
- test_download_cmems.py
- load_config
- argo_validation.py
- embedding_encoder.py
- underperformance_check.py
- test_depth_channel_and_units.py
- download_argo.py
- audit_config_compliance.py
- harmonize.py
- SurfaceOnlyDataset
- _run_epoch
- baselines/__init__.py
- dashboard/__init__.py
- models/__init__.py
- preprocessing/__init__.py
- validation/__init__.py
- tests/__init__.py
- OceanEmbed
- rules/graphify.md
- workflows/graphify.md
- _model_rmse

## God Nodes (most connected - your core abstractions)
1. `load_config()` - 55 edges
2. `run()` - 17 edges
3. `_subset_product_monthly()` - 14 edges
4. `_subset_product()` - 13 edges
5. `OceanEmbedReconstructionModel` - 13 edges
6. `harmonize()` - 12 edges
7. `test_loss_unit_consistency()` - 11 edges
8. `validate()` - 11 edges
9. `_get_date_range()` - 10 edges
10. `_generate_monthly_chunks()` - 10 edges

## Surprising Connections (you probably didn't know these)
- `main()` --calls--> `load_config()`  [EXTRACTED]
  scripts/00_test_access.py → src/config.py
- `main()` --calls--> `load_config()`  [EXTRACTED]
  scripts/01_test_harmonization.py → src/config.py
- `main()` --calls--> `load_config()`  [EXTRACTED]
  scripts/02_test_targets.py → src/config.py
- `main()` --calls--> `load_config()`  [EXTRACTED]
  scripts/03_test_splits.py → src/config.py
- `main()` --calls--> `load_config()`  [EXTRACTED]
  scripts/04_test_embedding.py → src/config.py

## Import Cycles
- None detected.

## Communities (26 total, 12 thin omitted)

### Community 0 - "pathlib"
Cohesion: 0.08
Nodes (45): argparse, cache_data, Figure, json, matplotlib_pyplot, numpy, pathlib, plotly_express (+37 more)

### Community 1 - "test_download_cmems.py"
Cohesion: 0.08
Nodes (57): cartopy_crs, os, pytest, _import_copernicusmarine(), main(), Path, Download and plot one configured day and tile of the configured SST product., _chunk_output_filename() (+49 more)

### Community 2 - "load_config"
Cohesion: 0.09
Nodes (33): copy, pandas, main(), Path, main(), Verification test suite for Phase 1: Environment & Data Access. Tests: 1.…, test_config_parameter_responsiveness(), test_logins() (+25 more)

### Community 3 - "argo_validation.py"
Cohesion: 0.12
Nodes (26): scipy_stats, main(), DataFrame, Path, Generate the OceanEmbed ARGO validation report and plots., _taylor_plot(), _empty_metric(), generate_predicted_field() (+18 more)

### Community 4 - "embedding_encoder.py"
Cohesion: 0.11
Nodes (15): math, ConvBlock, Tensor, CNN encoder and mirrored decoder for surface-field embeddings., U-Net-style convolutional encoder returning one vector per surface day., Return U-Net skip features followed by the bottleneck embedding., Decoder mirroring the encoder and restoring the configured grid size., SurfaceDecoder (+7 more)

### Community 5 - "underperformance_check.py"
Cohesion: 0.17
Nodes (25): datetime64, matplotlib_gridspec, matplotlib_ticker, _banner(), _check_channel_alignment(), _load_checkpoint(), _load_history(), main() (+17 more)

### Community 6 - "test_depth_channel_and_units.py"
Cohesion: 0.24
Nodes (19): _banner(), _fail(), _is_physical_celsius(), _is_zscore(), main(), _ok(), Any, ndarray (+11 more)

### Community 7 - "download_argo.py"
Cohesion: 0.16
Nodes (16): logging, requests, _build_erddap_url(), download_all(), download_argo_chunk(), _is_valid_argo_nc(), Any, Path (+8 more)

### Community 8 - "audit_config_compliance.py"
Cohesion: 0.32
Nodes (13): _audit_argo_summary(), _audit_source_literals(), _build_literal_patterns(), _check_cube(), _is_comment(), _line_uses_config(), main(), _mismatch() (+5 more)

### Community 9 - "harmonize.py"
Cohesion: 0.12
Nodes (29): gc, matplotlib, re, main(), _plot_variables(), Dataset, Path, Validate and plot the harmonized OceanEmbed Zarr dataset. (+21 more)

### Community 10 - "SurfaceOnlyDataset"
Cohesion: 0.16
Nodes (13): main(), Path, _available_indices(), Any, Dataset, Path, Tensor, Surface-only days with fast in-memory array slicing. (+5 more)

### Community 11 - "_run_epoch"
Cohesion: 0.20
Nodes (12): DataLoader, Optimizer, _depth_weights(), _loss(), Any, device, Module, ndarray (+4 more)

### Community 22 - "OceanEmbed"
Cohesion: 0.22
Nodes (8): Architecture, Configuration, Dashboard, OceanEmbed, Ordered reproduction, Outputs, Setup, Tests and hardcoding audit

### Community 25 - "_model_rmse"
Cohesion: 0.50
Nodes (4): OceanEmbedDataset, _model_rmse(), device, Module

## Knowledge Gaps
- **9 isolated node(s):** `graphify`, `Workflow: graphify`, `Architecture`, `Setup`, `Configuration` (+4 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 145 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **12 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `load_config()` connect `load_config` to `pathlib`, `test_download_cmems.py`, `argo_validation.py`, `underperformance_check.py`, `test_depth_channel_and_units.py`, `download_argo.py`, `audit_config_compliance.py`, `harmonize.py`, `SurfaceOnlyDataset`, `_run_epoch`?**
  _High betweenness centrality (0.288) - this node is a cross-community bridge._
- **Why does `OceanEmbedReconstructionModel` connect `test_depth_channel_and_units.py` to `pathlib`, `load_config`, `argo_validation.py`, `embedding_encoder.py`, `_run_epoch`?**
  _High betweenness centrality (0.041) - this node is a cross-community bridge._
- **Why does `SurfaceOnlyDataset` connect `SurfaceOnlyDataset` to `pathlib`?**
  _High betweenness centrality (0.031) - this node is a cross-community bridge._
- **What connects `graphify`, `Workflow: graphify`, `Architecture` to the rest of the system?**
  _9 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `pathlib` be split into smaller, more focused modules?**
  _Cohesion score 0.07547169811320754 - nodes in this community are weakly interconnected._
- **Should `test_download_cmems.py` be split into smaller, more focused modules?**
  _Cohesion score 0.07627118644067797 - nodes in this community are weakly interconnected._
- **Should `load_config` be split into smaller, more focused modules?**
  _Cohesion score 0.09102564102564102 - nodes in this community are weakly interconnected._