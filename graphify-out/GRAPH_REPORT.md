# Graph Report - SIH PROJECT  (2026-09-21)

## Corpus Check
- 41 files · ~21,213 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 4 file(s) not represented in the graph (top: (none) 2, .toml 1, .bat 1)

## Summary
- 392 nodes · 941 edges · 26 communities (14 shown, 12 thin omitted)
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 6 edges (avg confidence: 0.88)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `c2a4f1a6`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- config.py
- test_download_cmems.py
- load_config
- argo_validation.py
- OceanEmbedReconstructionModel
- underperformance_check.py
- test_depth_channel_and_units.py
- app.py
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
- main

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
  scripts/04_test_embedding.py → src/config.py
- `main()` --calls--> `SurfaceAutoencoder`  [EXTRACTED]
  scripts/04_test_embedding.py → src/models/embedding_encoder.py
- `main()` --calls--> `evaluate_climatology()`  [EXTRACTED]
  scripts/05_evaluate_training.py → src/baselines/climatology.py

## Import Cycles
- None detected.

## Communities (26 total, 12 thin omitted)

### Community 0 - "config.py"
Cohesion: 0.10
Nodes (37): argparse, copy, json, logging, matplotlib, matplotlib_pyplot, numpy, pandas (+29 more)

### Community 1 - "test_download_cmems.py"
Cohesion: 0.07
Nodes (55): cartopy_crs, pytest, _import_copernicusmarine(), main(), Path, Download and plot one configured day and tile of the configured SST product., _chunk_output_filename(), download_all() (+47 more)

### Community 2 - "load_config"
Cohesion: 0.10
Nodes (28): main(), _metric_line(), ndarray, Path, main(), Path, _report(), main() (+20 more)

### Community 3 - "argo_validation.py"
Cohesion: 0.08
Nodes (40): os, scipy_stats, shutil, _build_erddap_url(), download_all(), download_argo_chunk(), _is_valid_argo_nc(), Any (+32 more)

### Community 4 - "OceanEmbedReconstructionModel"
Cohesion: 0.10
Nodes (18): math, ConvBlock, Tensor, CNN encoder and mirrored decoder for surface-field embeddings., U-Net-style convolutional encoder returning one vector per surface day., Return U-Net skip features followed by the bottleneck embedding., Decoder mirroring the encoder and restoring the configured grid size., Composition of the independently usable surface encoder and decoder. (+10 more)

### Community 5 - "underperformance_check.py"
Cohesion: 0.17
Nodes (25): datetime64, matplotlib_gridspec, matplotlib_ticker, _banner(), _check_channel_alignment(), _load_checkpoint(), _load_history(), main() (+17 more)

### Community 6 - "test_depth_channel_and_units.py"
Cohesion: 0.27
Nodes (17): _banner(), _fail(), _is_physical_celsius(), _is_zscore(), main(), _ok(), Any, ndarray (+9 more)

### Community 7 - "app.py"
Cohesion: 0.14
Nodes (27): cache_data, Figure, plotly_express, plotly_graph_objects, main(), evaluate_climatology(), monthly_climatology(), DataArray (+19 more)

### Community 8 - "audit_config_compliance.py"
Cohesion: 0.29
Nodes (14): re, _audit_argo_summary(), _audit_source_literals(), _build_literal_patterns(), _check_cube(), _is_comment(), _line_uses_config(), main() (+6 more)

### Community 9 - "harmonize.py"
Cohesion: 0.15
Nodes (25): gc, main(), _plot_variables(), Dataset, Path, _report(), _coordinate_name(), _files_for_input() (+17 more)

### Community 10 - "SurfaceOnlyDataset"
Cohesion: 0.18
Nodes (11): main(), Path, _available_indices(), Any, Dataset, Path, Tensor, Surface-only days with fast in-memory array slicing. (+3 more)

### Community 11 - "_run_epoch"
Cohesion: 0.20
Nodes (12): DataLoader, Optimizer, _depth_weights(), _loss(), Any, device, Module, ndarray (+4 more)

### Community 22 - "OceanEmbed"
Cohesion: 0.22
Nodes (8): Architecture, Configuration, Dashboard, OceanEmbed, Ordered reproduction, Outputs, Setup, Tests and hardcoding audit

### Community 25 - "main"
Cohesion: 0.33
Nodes (6): OceanEmbedDataset, main(), _model_rmse(), device, Module, Path

## Knowledge Gaps
- **9 isolated node(s):** `graphify`, `Workflow: graphify`, `Architecture`, `Setup`, `Configuration` (+4 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 145 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **12 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `load_config()` connect `load_config` to `config.py`, `test_download_cmems.py`, `argo_validation.py`, `underperformance_check.py`, `test_depth_channel_and_units.py`, `app.py`, `audit_config_compliance.py`, `harmonize.py`, `SurfaceOnlyDataset`, `_run_epoch`, `main`?**
  _High betweenness centrality (0.285) - this node is a cross-community bridge._
- **Why does `OceanEmbedReconstructionModel` connect `OceanEmbedReconstructionModel` to `config.py`, `argo_validation.py`, `test_depth_channel_and_units.py`, `_run_epoch`, `main`?**
  _High betweenness centrality (0.040) - this node is a cross-community bridge._
- **Why does `SurfaceOnlyDataset` connect `SurfaceOnlyDataset` to `config.py`?**
  _High betweenness centrality (0.030) - this node is a cross-community bridge._
- **What connects `graphify`, `Workflow: graphify`, `Architecture` to the rest of the system?**
  _9 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `config.py` be split into smaller, more focused modules?**
  _Cohesion score 0.10272536687631027 - nodes in this community are weakly interconnected._
- **Should `test_download_cmems.py` be split into smaller, more focused modules?**
  _Cohesion score 0.07268170426065163 - nodes in this community are weakly interconnected._
- **Should `load_config` be split into smaller, more focused modules?**
  _Cohesion score 0.09852216748768473 - nodes in this community are weakly interconnected._