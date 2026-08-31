# Data licenses & redistribution

This repository re-hosts datasets and derived files **for internal reproduction only**. Several carry
**no-derivatives / non-commercial / challenge** terms that do **not** permit redistribution. Keep this repo private;
for any use beyond reproducing the paper internally, obtain each dataset from its official source under its own terms.

## Raw datasets (in the private Release; do NOT redistribute)

| Dataset | Used for | License / terms | Official source |
|---|---|---|---|
| **Harvard-GDP** | thickness-map axis | CC BY-NC-**ND** 4.0 (non-commercial, **no derivatives**) | HuggingFace `harvardairobotics/Harvard-GDP` |
| **Harvard-GF** | map-axis extension | CC BY-NC-**ND** 4.0 | HF mirror; Harvard Dataverse (gated, ToU + `DATAVERSE_TOKEN`) |
| **GLEAM** | Spectralis B-scan axis | CC BY-NC-**ND**; single-center, available **on request** | request from the data owners |
| **GOALS** | cross-vendor (Topcon) validity | MICCAI-2022 challenge terms of use | Baidu AI Studio competition/230 (iChallenge-GON3) |
| **RETOUCH** | speckle calibration | MICCAI-2017 challenge terms of use | `retouch.grand-challenge.org` (registration) |

None of these may be re-hosted publicly or redistributed; the copies in this repo's Release exist solely so Ufonia
colleagues can reproduce the paper without re-downloading. **Non-commercial + no-derivatives applies** — do not use the
raw images for product/commercial work.

## Derived files committed in git (`data/derived/*.npz`)

Per-eye model **outputs** (probabilities / cached features), several steps removed from the source images — materially
lower risk than the raw images, shipped to make the figures reproducible without re-running any model:

- `{gemini,openai,claude,qwen,medgemma}_*_contender.npz` and `map_fm/*` — per-eye VLM decision probabilities.
- `eyened_cprnfl.npz` — an extracted cpRNFL thickness feature computed from **GLEAM** B-scans; inherits GLEAM's
  CC BY-NC-ND (no-derivatives) restriction. Treat as internal-only.

No PHI is present in any committed file (only aggregate counts and de-identified sequential IDs).

## Code

The code in `scripts/` and `src/` is released under the **MIT License** ([LICENSE](LICENSE)). The MIT grant covers the
**code only** — it does **not** relicense the datasets or the derived data above, which remain under their own terms.
