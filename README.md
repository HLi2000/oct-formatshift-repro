# OCT-FormatShift — reproducibility package

Everything needed to reproduce the OCT-FormatShift paper — *When the Scanner Changes: Calibrated, Decision-Level
Reliability of Frontier VLMs and Foundation Models for Glaucoma Triage* — end to end: the paper source, the analysis
code, the committed result files, and (via a private Release) the large per-eye outputs and raw datasets.

> **Private / internal only.** This repo re-hosts datasets under CC BY-NC-ND and challenge terms of use (GLEAM, GOALS,
> RETOUCH, Harvard-GDP/GF) purely for internal reproduction. **Do not make it public and do not redistribute the raw
> data.** See [DATA_LICENSES.md](DATA_LICENSES.md). Code is MIT ([LICENSE](LICENSE)); the data is not.

## Layout
```
papers/oct-formatshift-vlm4rwd/   the paper (self-contained LaTeX: main.tex + figures/)
scripts/                          build_paper_tables, build_paper_submission_figures, analyze_shift_mechanism, fetch_oct_data
src/glaucoma_triage/              the analysis library (oct_format/*)
reports/*.json                    committed panels/metrics — every NUMBER reproduces from these
data/derived/*.npz                committed per-eye VLM outputs (~0.8 MB) + cross-vendor cpRNFL
```

## Install
```bash
uv venv && uv pip install -e .        # or: python -m venv .venv && .venv/bin/pip install -e .
```
Slim, torch-free dependency set (numpy/scipy/scikit-learn/matplotlib/pandas/pyyaml/pillow).

## Reproduce — three tiers

**1. Every number (offline, no download).** All tables + the scatter/ρ values in Fig 3 come from the committed
`reports/*.json`:
```bash
python scripts/build_paper_tables.py            # regenerates the paper's table .tex files
```

**2 & 3. All figures (needs the Release assets).** The figure builder rebuilds the raw-image figures (Fig 1, A1, A2),
the per-eye reliability figure (A3), and the JSON-driven ones (Fig 2, 3, A4, A5) in one pass, so it needs the large
binaries first. Fetch + verify them against the manifest, unpack, then build:
```bash
export GH_TOKEN=...                              # a token with read access to this private repo
python scripts/fetch_oct_data.py                # downloads + SHA256-verifies the 5 Release assets, prints unpack cmds
#   -> run the printed: tar -xf ... ; unzip gleam/goals ; cat RETOUCH_part_aa RETOUCH_part_ab > RETOUCH.tar
python scripts/build_paper_submission_figures.py # all 11 figure PDFs -> papers/oct-formatshift-vlm4rwd/figures/
```
`data/derived/*.npz` in git are the frontier-VLM outputs (Fig 2/3 curves); the larger foundation-model / CNN npz and
the raw datasets live in the Release, pinned by SHA256 in `reports/oct_format_data_manifest.json`.

## Build the paper
```bash
cd papers/oct-formatshift-vlm4rwd && tectonic main.tex     # figures are already committed, so this works as-is
```

## What's here vs. not
- **Here**: the paper, the paper-build pipeline, committed results, and (via Release) the per-eye npz + raw datasets.
- **Not here**: the contender/panel-*generation* stack (VLM API calls, GPU foundation-model feature extraction) that
  produced `data/derived/*` and `reports/*` from raw images — that needs API keys, a GPU, and torch, and lives in the
  authors' full working repo. This package consumes those committed outputs; it does not re-run the models.

## Data
Fetched into `data/` by `fetch_oct_data.py`. Every dataset is CC BY-NC-ND / challenge-gated — obtain from the official
source for any use beyond internal reproduction. Public sources and per-dataset terms: [DATA_LICENSES.md](DATA_LICENSES.md).
