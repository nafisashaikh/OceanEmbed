# Graph Report - SIH PROJECT  (2026-09-20)

## Corpus Check
- 41 files · ~19,180 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 5 file(s) not represented in the graph (top: (none) 3, .bat 1, .nc 1)

## Summary
- 383 nodes · 910 edges · 23 communities (11 shown, 12 thin omitted)
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
- SurfaceEncoder
- underperformance_check.py
- test_depth_channel_and_units.py
- audit_config_compliance.py
- harmonize.py
- 04_test_embedding.py
- baselines/__init__.py
- dashboard/__init__.py
- models/__init__.py
- preprocessing/__init__.py
- validation/__init__.py
- tests/__init__.py
- OceanEmbed
- rules/graphify.md
- workflows/graphify.md

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
- `main()` --calls--> `harmonize()`  [EXTRACTED]
  scripts/01_test_harmonization.py → src/preprocessing/harmonize.py
- `main()` --calls--> `load_config()`  [EXTRACTED]
  scripts/02_test_targets.py → src/config.py
- `main()` --calls--> `load_config()`  [EXTRACTED]
  scripts/03_test_splits.py → src/config.py

## Import Cycles
- None detected.

## Communities (23 total, 12 thin omitted)

### Community 0 - "pathlib"
Cohesion: 0.08
Nodes (43): argparse, cache_data, Figure, matplotlib, matplotlib_pyplot, numpy, pandas, pathlib (+35 more)

### Community 1 - "test_download_cmems.py"
Cohesion: 0.08
Nodes (57): cartopy_crs, os, pytest, _import_copernicusmarine(), main(), Path, Download and plot one configured day and tile of the configured SST product., _chunk_output_filename() (+49 more)

### Community 2 - "load_config"
Cohesion: 0.08
Nodes (40): logging, requests, main(), Verification test suite for Phase 1: Environment & Data Access. Tests: 1.…, test_config_parameter_responsiveness(), test_logins(), test_sanity_sst(), test_six_datasets() (+32 more)

### Community 3 - "argo_validation.py"
Cohesion: 0.10
Nodes (31): scipy_stats, main(), DataFrame, Path, _taylor_plot(), evaluate_climatology(), monthly_climatology(), DataArray (+23 more)

### Community 4 - "SurfaceEncoder"
Cohesion: 0.13
Nodes (10): ConvBlock, Tensor, U-Net-style convolutional encoder returning one vector per surface day., Return U-Net skip features followed by the bottleneck embedding., Decoder mirroring the encoder and restoring the configured grid size., SurfaceDecoder, SurfaceEncoder, Tensor (+2 more)

### Community 5 - "underperformance_check.py"
Cohesion: 0.17
Nodes (25): datetime64, matplotlib_gridspec, matplotlib_ticker, _banner(), _check_channel_alignment(), _load_checkpoint(), _load_history(), main() (+17 more)

### Community 6 - "test_depth_channel_and_units.py"
Cohesion: 0.24
Nodes (19): _banner(), _fail(), _is_physical_celsius(), _is_zscore(), main(), _ok(), Any, ndarray (+11 more)

### Community 8 - "audit_config_compliance.py"
Cohesion: 0.32
Nodes (13): _audit_argo_summary(), _audit_source_literals(), _build_literal_patterns(), _check_cube(), _is_comment(), _line_uses_config(), main(), _mismatch() (+5 more)

### Community 9 - "harmonize.py"
Cohesion: 0.17
Nodes (22): gc, re, shutil, _coordinate_name(), _files_for_input(), _group_files_by_month(), harmonize(), _month_key_from_path() (+14 more)

### Community 10 - "04_test_embedding.py"
Cohesion: 0.05
Nodes (47): copy, DataLoader, json, math, OceanEmbedDataset, Optimizer, random, main() (+39 more)

### Community 22 - "OceanEmbed"
Cohesion: 0.22
Nodes (8): Architecture, Configuration, Dashboard, OceanEmbed, Ordered reproduction, Outputs, Setup, Tests and hardcoding audit

## Knowledge Gaps
- **9 isolated node(s):** `graphify`, `Workflow: graphify`, `Architecture`, `Setup`, `Configuration` (+4 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 145 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **12 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `load_config()` connect `load_config` to `pathlib`, `test_download_cmems.py`, `argo_validation.py`, `underperformance_check.py`, `test_depth_channel_and_units.py`, `audit_config_compliance.py`, `harmonize.py`, `04_test_embedding.py`?**
  _High betweenness centrality (0.290) - this node is a cross-community bridge._
- **Why does `OceanEmbedReconstructionModel` connect `test_depth_channel_and_units.py` to `04_test_embedding.py`, `argo_validation.py`, `SurfaceEncoder`?**
  _High betweenness centrality (0.041) - this node is a cross-community bridge._
- **What connects `graphify`, `Workflow: graphify`, `Architecture` to the rest of the system?**
  _9 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `pathlib` be split into smaller, more focused modules?**
  _Cohesion score 0.07510204081632653 - nodes in this community are weakly interconnected._
- **Should `test_download_cmems.py` be split into smaller, more focused modules?**
  _Cohesion score 0.07627118644067797 - nodes in this community are weakly interconnected._
- **Should `load_config` be split into smaller, more focused modules?**
  _Cohesion score 0.07928118393234672 - nodes in this community are weakly interconnected._
- **Should `argo_validation.py` be split into smaller, more focused modules?**
  _Cohesion score 0.10227272727272728 - nodes in this community are weakly interconnected._