# nocturnal-moist-heat-burden

Code, frozen result artefacts and figure pipeline for:

> **Night-time moist heat driven by humidity, not temperature alone, is crossing
> physiologically anchored survival thresholds — with mortality risk attaching to the
> crossing event itself.**
> Manuscript prepared for *Nature Climate Change* (initial submission, anonymised).

[![Release](https://img.shields.io/badge/release-v1.0.0-blue)](https://github.com/fanqimcse/nocturnal-moist-heat-burden/releases/tag/v1.0.0)
Archival request accepted by Software Heritage (SWHID to be minted after the first visit).

## What this repository contains

| Path | Content |
|---|---|
| `code/` | Full analysis pipeline, steps 1–54 (`step*.py`), the shared toolkit `ghcnh_toolkit.py`, ingestion (`ghcnh_fast.py`) and unit tests |
| `results/` | The **67 frozen result artefacts** (`*.json` + companion `*.txt` reports) that back every number in the manuscript |
| `audit/` | `baseline_hash.json` (SHA-256 index of the frozen artefacts), `rerun_core.sh` and `rerun_summary.txt` (14/14 byte-identical re-run audit) |
| `manuscript/` | LaTeX source of the Article, Extended Data captions/table, Supplementary Information and `references.bib` |
| `figures/` | Main Figures 1–5 and Extended Data Figures 1–6 (vector PDF + SVG) plus `figures_manifest.csv` (450 traced display values) |
| `tables/` | Main results table (`main_results_table.csv/.tex`) |
| `references/` | Citation-evidence table (quote ↔ manuscript claim, per-reference verification status), download manifest and report |
| `docs_import/` | Internal audit trail: reproduction & literature audit, novelty assessment, peer-review record (Chinese) |

**Not included** (by design): raw station archives (`ghcn_data/`, ~GB scale) and third-party
external datasets (`ext/`); all are public and listed in the manuscript's Data Availability
statement. Downloaded publisher PDFs of cited literature are excluded to respect copyright; the
download/verification metadata are in `references/`.

## Reproducibility

Every displayed number in the manuscript traces to a hash-verified artefact:

1. `audit/baseline_hash.json` stores a SHA-256 (normalised-JSON) for each of the 67 artefacts;
   `make_figures.py` recomputes them and aborts on any mismatch.
2. `audit/rerun_core.sh` re-ran the 14 core scripts end-to-end; all 14 reproduced
   **byte-identical** results (total 25 min 36 s; see `audit/rerun_summary.txt`).
3. Rebuild all figures and tables (deterministic; two consecutive runs produce identical SHA-256):

```bash
python manuscript/make_figures.py          # requires matplotlib + numpy
```

4. Retracted/failed analyses are registered, not hidden: Extended Data Table 1 rows R1–R6 and
   `docs_import/研究路径复现审计与文献对照报告.md`.

## Data sources (all public)

NOAA GHCNh v1.1.0 · Eurostat weekly deaths · NASA POWER (MERRA-2) · Open-Meteo/ERA5 ·
Hong Kong C&SD Table 115-01022A · SingStat M810121 · CMIP6/DAMIP via ESGF · HadEX3 · GISTEMP v4 ·
GeoNames · WHO Mortality Database.

## Headline results

- Global night-time hours above the 21.9 °C wet-bulb survivability bound: **+42.84 h per decade**
  (1976–2023, n = 64 stations); positive in all eight analysis windows.
- Warming alone explains **−6%**; humidity change required for **+106%**.
- Optimal-fingerprint scaling factor **b_ALL = 0.429 ± 0.148** (NAT rejected); CMIP6 overstates the
  rise 2.3–2.9×.
- Europe (225 regions, 77,551 region-weeks): crossing weeks RR = **1.0181** [1.0104, 1.0258];
  risk attaches to the crossing event (step-shaped dose–response); intensity-based scaling
  overstates risk-relevant growth **1.78×**.
- Wet-tropics replication (Hong Kong + Singapore, orthogonalised meta): **1.0030**
  [1.0009, 1.0051], I² = 0%; dry-bulb negative control is inverse.
- Percentile-based warning triggers fail under zero inflation (P95 = 0 in **75.6%** of regions).

## Requirements

Python ≥ 3.11; `numpy`, `pandas`, `scipy`, `statsmodels`, `xarray`, `matplotlib`, `netCDF4`.
No proprietary software is used.

## License

MIT for code. Result artefacts and manuscript text: CC-BY 4.0.
