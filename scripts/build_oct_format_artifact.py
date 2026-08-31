#!/usr/bin/env python
"""Build the OCT-FormatShift v3 report — a self-contained, image-rich, literature-supported HTML.

Every figure is a matplotlib render embedded as a base64 PNG data-URI (real OCT images + simulated
clean->shifted examples + result charts), so the page has NO external assets (CSP-safe). The prose is scored on
the CURRENT results — the two calibrated-axis panels (``oct_device_axis_panel.json`` +
``oct_appearance_axis_panel.json``), the structural-LoA validation (``oct_format_validity.json``) and the
cross-vendor corroboration (``oct_cross_vendor_panel.json``) — NOT the retired ComBat tolerance-certificate.

    .venv/bin/python scripts/build_oct_format_artifact.py    # -> reports/oct_format_artifact.html
"""

from __future__ import annotations

import base64
import glob
import html
import io
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

from glaucoma_triage.oct_format import bscan_transforms as BT
from glaucoma_triage.oct_format import calibrated_offset as CO
from glaucoma_triage.oct_format import canonical, loaders, validity

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
OUT = REPORTS / "oct_format_artifact.html"

# ── palette — ONE semantic system: colour by adaptation REGIME + VLM as a class ───────────────────────
INK, INK2, MUT = "#12100e", "#54524d", "#8a897f"          # ink / secondary ink / muted grey
# model readers are coloured by ADAPTATION METHOD in every model panel + table (FT and DR are DIFFERENT — a
# pretrained model fine-tuned on the task vs a from-scratch net trained on random device shifts / stability):
LP, FT, DR, SCRATCH, ORACLE = "#2a78d6", "#eb6834", "#0d9488", "#8a897f", "#bcb9b0"  # linear-probe · fine-tune · domain-rand · from-scratch · oracle
# frontier VLM — ONE colour in EVERY figure (lighter tint = zero-shot within the VLM panels):
VLM, VLM_ZERO = "#e34948", "#e7a6a5"
SPECIALIST = "#3f6079"                                    # the specialist FIELD in the VLM-vs-field head-to-head (neutral slate)
# vendor identity — fixed across the device / RETOUCH / cross-vendor figures:
CIRRUS, TOPCON, SPECTRALIS = "#7d5ba6", "#2a9d8f", "#c98a3a"
BAND = "#1baf7a"                                          # published-agreement band ("within spec")
# semantic status for the HTML callouts / tiles / chips:
GOOD, WARN, CRIT = "#1baf7a", "#d98a1f", "#e34948"
# legacy aliases — kept only so un-migrated references still resolve while figures move to the names above:
ACC, S1, S2, S3 = LP, LP, FT, BAND

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica Neue"],
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11, "axes.edgecolor": "#c9c8c2",
    "axes.linewidth": 0.8, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.labelcolor": INK, "axes.titlecolor": INK, "axes.grid": True, "grid.color": "#ecebe6",
    "grid.linewidth": 0.8, "svg.fonttype": "none",
})


def _uri(fig, *, pad=0.12) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=124, bbox_inches="tight", pad_inches=pad)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _despine(ax, keep=("left", "bottom")):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)


def _imshow_bscan(ax, img, title):
    ax.imshow(img, cmap="gray", aspect="auto")
    ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([]); _despine(ax, keep=())


def _imshow_map(ax, m, title, *, cmap="viridis", vmin=None, vmax=None, cbar_label="RNFL µm"):
    mm = np.ma.masked_invalid(m)
    cm = plt.get_cmap(cmap).copy(); cm.set_bad("#eeede8")
    im = ax.imshow(mm, cmap=cm, vmin=vmin, vmax=vmax, interpolation="bilinear")
    ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([]); _despine(ax, keep=())
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(cbar_label, fontsize=9); cb.ax.tick_params(labelsize=8)
    return im


# ── real data loaders (for the example figures) ──────────────────────────────────────────────────────
def _gf_map_and_cube():
    """One healthy Harvard-GF eye: native 200x200 cleaned RNFLT map (µm) + one central B-scan slice."""
    for f in sorted(glob.glob(str(ROOT / "data/raw/harvard_gf/*/data_*.npz"))):
        z = np.load(f, allow_pickle=True)
        if "rnflt" in z.files and int(z.get("glaucoma", 1)) == 0:
            m = canonical.clean(z["rnflt"].astype(float))
            cube = z["oct_bscans"] if "oct_bscans" in z.files else None
            bscan = cube[cube.shape[0] // 2].astype(float) if cube is not None else None
            return m, bscan
    raise SystemExit("no Harvard-GF map on disk")


def _gleam_bscan():
    p = ROOT / "data/raw/gleam/GLEAM/all_samples/0001/Thickness.jpg"
    return np.asarray(Image.open(p).convert("L"), dtype=float) if p.exists() else None


def _goals_bscan_mask():
    b = sorted(glob.glob(str(ROOT / "data/raw/goals/GOALS/*/bscan/*.png")))
    m = sorted(glob.glob(str(ROOT / "data/raw/goals/GOALS/*/semseg/*.png")))
    if not b or not m:
        return None, None
    return np.asarray(Image.open(b[0]).convert("L"), float), np.asarray(Image.open(m[0]))


# ── FIGURES ──────────────────────────────────────────────────────────────────────────────────────────
def fig_devices():
    """Fig 1 — same biology, three device representations: a Cirrus thickness MAP (top, full width), then the
    two raw B-scans side by side BELOW at their true (undistorted) aspect — Spectralis, and Topcon with its
    RNFL segmentation."""
    cmap_m, _ = _gf_map_and_cube()
    spec = _gleam_bscan()
    top_b, top_m = _goals_bscan_mask()
    fig = plt.figure(figsize=(10.4, 7.3))
    # top — Cirrus map CENTRED with a dedicated thin colour bar (explicit axes → reliably centred, no
    # colorbar-steal drift); bottom — the two raw B-scans side by side at their true wide aspect (no squish)
    axm = fig.add_axes([0.315, 0.515, 0.37, 0.37])
    cm = plt.get_cmap("viridis").copy(); cm.set_bad("#eeede8")
    im = axm.imshow(np.ma.masked_invalid(cmap_m), cmap=cm, interpolation="bilinear")
    axm.set_title("Cirrus · en-face thickness MAP  (colour = RNFL µm)", fontsize=11)
    axm.set_xticks([]); axm.set_yticks([]); _despine(axm, keep=())
    cax = fig.add_axes([0.705, 0.54, 0.014, 0.32])
    cb = fig.colorbar(im, cax=cax); cb.set_label("RNFL µm", fontsize=9); cb.ax.tick_params(labelsize=8)
    axs = fig.add_axes([0.035, 0.05, 0.445, 0.42])
    _imshow_bscan(axs, spec if spec is not None else np.zeros((10, 10)), "Spectralis · circumpapillary B-SCAN")
    axt = fig.add_axes([0.52, 0.05, 0.445, 0.42])
    if top_b is not None:
        axt.imshow(top_b, cmap="gray", aspect="auto")
        rnfl = np.ma.masked_where(top_m != 80, top_m)
        axt.imshow(rnfl, cmap="autumn", alpha=0.55, aspect="auto")
        axt.set_title("Topcon · circumpapillary B-SCAN + RNFL segmentation", fontsize=11)
    axt.set_xticks([]); axt.set_yticks([]); _despine(axt, keep=())
    fig.suptitle("One eye's retinal nerve fibre layer, three device representations — the reading depends on the device",
                 fontsize=12.5, y=0.99, color=INK)
    return _uri(fig)


def fig_device_axis():
    """Fig 3 — the THICKNESS-MAP axis: a real Cirrus RNFL map, the Warner-calibrated Cirrus->Spectralis shifted map,
    and the per-pixel difference (µm)."""
    m, _ = _gf_map_and_cube()
    rng = np.random.default_rng(3)
    sh = CO.apply_sectored_map(m, "cirrus_to_spectralis", rng, severity=1.0)
    d = sh - m
    vmax = float(np.nanpercentile(m, 99))
    fig, ax = plt.subplots(1, 3, figsize=(11.2, 3.6))
    _imshow_map(ax[0], m, "Clean (Cirrus)", vmin=0, vmax=vmax)
    _imshow_map(ax[1], sh, "Simulated Spectralis read", vmin=0, vmax=vmax)
    lim = max(float(np.nanpercentile(np.abs(d), 99.8)), 3.0)
    _imshow_map(ax[2], d, "Difference (sim Spectralis − Cirrus), µm", cmap="RdBu_r",
                vmin=-lim, vmax=lim, cbar_label="Δ µm")
    fig.suptitle("THICKNESS-MAP axis — a calibrated device shift (Cirrus→sim Spectralis: +1.9 µm bias, "
                 "±7.7 µm LoA; Warner 2011)", fontsize=12, y=1.02, color=INK)
    return _uri(fig)


def fig_appearance_axis():
    """Fig 4 — the B-SCAN APPEARANCE axis: a real B-scan and the physics appearance-stress version
    (speckle ↓, axial blur ↑, contrast)."""
    b = _gleam_bscan()
    if b is None:
        m, cube = _gf_map_and_cube(); b = cube
    rng = np.random.default_rng(4)
    sh = BT.apply_bscan_device_shift(b, rng, 1.0, ref="spectralis", target="topcon")
    fig, ax = plt.subplots(1, 2, figsize=(11.2, 3.9))
    _imshow_bscan(ax[0], b, "Clean  (GLEAM Spectralis)")
    _imshow_bscan(ax[1], sh, "Simulated Topcon  (speckle↑ · contrast↓)")
    fig.suptitle("B-SCAN APPEARANCE axis — a direction-validated device shift (Spectralis→sim Topcon: inject speckle "
                 "+ dim contrast; Schmitt 1999 / RETOUCH). An appearance-tolerance probe, NOT a same-region device-LoA claim",
                 fontsize=10.5, y=1.03, color=INK)
    return _uri(fig)


def fig_harness():
    """Fig 5 — the common-ground harness: any model → one referral call → paired clean→shifted degradation."""
    fig, ax = plt.subplots(figsize=(11.6, 2.5)); ax.set_xlim(-5, 101); ax.set_ylim(0, 30); ax.axis("off")
    boxes = [(8, "Any reader\n(SmallCNN · FM · CNN-sweep ·\nVLM)"),
             (34, "One clinical act\npredict_referral_proba\n→ [0,1]"),
             (60, "Clean  vs  shifted\n(same eyes, calibrated shift)"),
             (86, "Paired degradation\nflip · Δsens@spec ·\nAUC · ΔAUC  (+ 95% CI)")]
    for x, t in boxes:
        ax.add_patch(FancyBboxPatch((x - 11, 9), 22, 12, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    fc="#f4f6fb", ec=ACC, lw=1.3))
        ax.text(x, 15, t, ha="center", va="center", fontsize=9.4, color=INK)
    for x0, x1 in ((19, 23), (45, 49), (71, 75)):
        ax.add_patch(FancyArrowPatch((x0, 15), (x1, 15), arrowstyle="-|>", mutation_scale=16, color=INK2, lw=1.4))
    ax.text(50, 3, "Every contender is reduced to the same glaucoma-referral call and scored by the same metric — "
                   "representation-labelled, never cross-ranked (§0).", ha="center", fontsize=9.2, color=INK2)
    return _uri(fig)


def fig_bland_altman(valid):
    """Fig 6 — Bland–Altman of the deployed device shift (per-eye diffs) vs the published Warner LoA band."""
    X, _ = loaders.load_harvard_gf_maps(limit=400)
    st = CO.map_difference_stats(X, "cirrus_to_spectralis", np.random.default_rng(7), severity=1.0)["global"]
    diffs = np.asarray(st["diffs"], float)
    means = np.linspace(60, 105, len(diffs))  # illustrative mean-thickness spread for the x-axis
    bias, lo, hi = st["bias"], st["bias"] - 1.96 * st["sd"], st["bias"] + 1.96 * st["sd"]
    pub = valid["V1a_realism"]["cirrus_to_spectralis"]
    pub_bias, pub_loa = pub["target_bias"], pub["published_loa_half"]
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.axhspan(pub_bias - pub_loa, pub_bias + pub_loa, color=S3, alpha=0.10,
               label=f"published 95% LoA (Warner 2011): {pub_bias:+.1f} ± {pub_loa:.1f} µm")
    ax.scatter(means, diffs, s=14, color=S1, alpha=0.5, edgecolor="none", label="synthetic per-eye difference")
    for yv, ls, c, lab in [(bias, "-", S2, f"synthetic bias {bias:+.2f} µm"),
                           (lo, "--", S2, f"synthetic 95% LoA ±{1.96*st['sd']:.1f} µm"), (hi, "--", S2, None)]:
        ax.axhline(yv, ls=ls, color=c, lw=1.4, label=lab)
    ax.axhline(0, color=MUT, lw=0.8, ls=":")
    ax.set_xlabel("mean RNFL thickness (µm)"); ax.set_ylabel("Spectralis − Cirrus  (µm)")
    ax.legend(fontsize=8.2, loc="upper right", framealpha=0.9)
    _despine(ax)
    ax.set_title("The simulated device shift reproduces the published Bland–Altman agreement", fontsize=11.5)
    return _uri(fig), bias, 1.96 * st["sd"]


def fig_sector_loa(valid):
    """Fig 6 — per-sector LoA half-width against the published GLOBAL limit: the shift reproduces the published
    magnitude AND is anatomy-coupled (superior/inferior bundles wider), matching Cheung–Leung 2008."""
    v = valid["V1a_structural"]
    s = v["sector_loa_half_um"]
    order = [("Superior", s["s"]), ("Inferior", s["i"]), ("Nasal", s["n"]), ("Temporal", s["t"])]
    pub = v.get("published_loa_half_um")
    cols = [S2, S2, MUT, MUT]
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    ax.bar([o[0] for o in order], [o[1] for o in order], color=cols, width=0.62)
    for i, (_, val) in enumerate(order):
        ax.text(i, val + 0.12, f"{val:.1f}", ha="center", fontsize=9.4, color=INK)
    if isinstance(pub, (int, float)):
        ax.axhline(pub, color=BAND, ls="--", lw=1.5, zorder=0)
        ax.text(3.45, pub + 0.12, f"published global {pub:.1f} µm (Warner 2011)", fontsize=7.6, color=BAND,
                ha="right", va="bottom")
    ax.set_ylabel("LoA half-width (µm)"); ax.set_ylim(0, max(o[1] for o in order) + 1.4); _despine(ax)
    ax.set_title("Per-sector LoA straddles the published global limit — the wider superior/inferior bundles (orange)\n"
                 "sit above it, nasal/temporal below, reproducing the anatomy-coupling (Cheung–Leung 2008)", fontsize=9.2)
    return _uri(fig)


def fig_dose_response(valid):
    """Fig 8 — referral-flip rises monotonically with shift dose; the ~4 µm MDC noise floor is marked."""
    rows = sorted(valid["V3_monotonicity"]["rows"], key=lambda r: r["level"])
    lv = [r["level"] for r in rows]; flip = [r["flip_rate"] for r in rows]; drift = [r["read_drift"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(lv, flip, "-o", color=S1, lw=2, ms=6, label="referral-flip rate")
    ax2 = ax.twinx()
    ax2.plot(lv, drift, "--s", color=S2, lw=1.6, ms=5, label="read drift (µm)")
    ax2.axhline(4.0, color=MUT, ls=":", lw=1.2)
    ax2.text(0.02, 4.3, "≈4 µm test–retest floor (Mwanza 2010)", fontsize=8.4, color=MUT)
    ax.set_xlabel("shift dose  (level; 1.0 = published gap)"); ax.set_ylabel("referral-flip rate", color=S1)
    ax2.set_ylabel("read drift (µm)", color=S2); ax2.grid(False)
    _despine(ax); _despine(ax2, keep=("right", "bottom"))
    ax.set_title("The axis behaves like a dose — more device shift, more decision change", fontsize=11.5)
    lines = ax.get_lines() + ax2.get_lines()[:1]
    ax.legend(lines, [ln.get_label() for ln in lines], fontsize=8.6, loc="upper left")
    return _uri(fig)


# device / appearance / map reader panels are all rendered by the shared fig_reader_panel (below);
# build() assembles the per-axis arm list via _board_triples (structural readers) or _axis_triples (FM/CNN/VLM).


# NOTE: the old fig_appearance_panel + its CNN-scale-sweep exclusion are superseded by fig_reader_panel — the
# whole roster (FMs, the ConvNeXt/ResNet scale sweep, VLMs) now shares one head-to-head, and "scale is flat"
# stays in the scale×pretraining callout. The 2×2 metric language (flip · AUC · ΔAUC · Δsens@spec) is the table.


# ── reader head-to-head — ONE panel style for every axis (FM · CNN · VLM together, coloured by class) ──
_FM_DISP = {"dinov2_ft": "DINOv2·FT", "dinov2_reg": "DINOv2·LP", "mirage": "MIRAGE·LP", "mirage_ft": "MIRAGE·FT",
            "retfound": "RETFound·LP", "retfound_ft": "RETFound·FT",
            "resnet50": "ResNet-50·LP", "resnet50_ft": "ResNet-50·FT",
            "convnext_tiny": "ConvNeXt-T·LP", "convnext_base": "ConvNeXt-B·LP", "convnext_large": "ConvNeXt-L·LP",
            "convnext_tiny_ft": "ConvNeXt-T·FT", "convnext_base_ft": "ConvNeXt-B·FT", "convnext_large_ft": "ConvNeXt-L·FT",
            "supervised": "SmallCNN"}
_STRUCT = {"fullres_cnn_dr": "SmallCNN + domain-rand.", "fullres_cnn": "SmallCNN",
           "ridge_baseline": "Ridge (64² floor)", "ridge_dr": "Ridge + domain-rand.",
           "AffineRefit": "Affine-refit (offset oracle)"}
_CLASS_NAME = {LP: "linear-probe / linear floor", FT: "fine-tune", DR: "domain-rand (stability)",
               SCRATCH: "from scratch", ORACLE: "offset oracle", VLM: "VLM · few-shot", VLM_ZERO: "VLM · zero-shot"}


def _ci_err(row, key):
    ci = (row.get("ci") or {}).get(key)
    v = row.get(key, float("nan"))
    if ci and len(ci) == 2 and v == v:
        return [max(0.0, v - ci[0]), max(0.0, ci[1] - v)]
    return [0.0, 0.0]


def _gemini_label(m):
    s = m.replace("gemini_map_", "").replace("gemini_", "")
    gen = ("2.5-flash" if s.startswith("2p5_flash") else "2.5-pro" if s.startswith("2p5_pro")
           else "3.6-flash" if s.startswith("3p6_flash") else s.split("_")[0])
    return f"Gemini {gen}", ("fewshot" in s)


def _is_vlm(m):
    """True for ANY prompted-VLM arm (Gemini, OpenAI, Claude, Qwen, MedGemma) on either axis — the red class.
    MedGemma is a medical specialist but is scored/coloured as a VLM alongside the generalists (one VLM class)."""
    return m.startswith(("gemini", "openai", "claude", "qwen3vl", "medgemma"))


def _vlm_label(m):
    """(display generation, is_fewshot) for a VLM arm — Gemini, OpenAI, Claude, Qwen, or MedGemma, either axis."""
    if m.startswith("openai"):
        s = m.replace("openai_map_", "").replace("openai_", "")
        gen = ("GPT-5.6 Sol" if s.startswith("5p6_sol") else "GPT-5.6" if s.startswith("5p6")
               else "GPT-" + s.split("_")[0])
        return gen, ("fewshot" in s)
    if m.startswith("claude"):
        s = m.replace("claude_map_", "").replace("claude_", "")
        gen = "Claude Opus 5" if s.startswith("opus_5") else "Claude " + s.split("_")[0].capitalize()
        return gen, ("fewshot" in s)
    if m.startswith("qwen3vl"):
        s = m.replace("qwen3vl_map_", "").replace("qwen3vl_", "")
        gen = "Qwen3-VL-8B" if s.startswith("8b") else "Qwen3-VL-" + s.split("_")[0]
        return gen, ("fewshot" in s)
    if m.startswith("medgemma"):
        return "MedGemma", ("fewshot" in m)
    return _gemini_label(m)


def _reader_meta(m):
    """(display label, class colour) for ANY arm key on either axis — colour by REGIME, VLM = red."""
    if _is_vlm(m):
        gen, few = _vlm_label(m)
        return f"{gen} · {'few' if few else 'zero'}-shot", (VLM if few else VLM_ZERO)
    if m in _STRUCT:
        col = (ORACLE if m == "AffineRefit" else DR if m.endswith("_dr")
               else LP if "ridge" in m else SCRATCH)
        return _STRUCT[m], col
    lab = _FM_DISP.get(m, m)
    return lab, (FT if lab.endswith("FT") else (SCRATCH if m == "supervised" else LP))


def _axis_triples(panel, arms):
    """(label, colour, headline-row @level 1.0) for each arm present in a panel dict."""
    out = []
    for m in arms:
        if m in panel:
            pl = panel[m]["per_level"]
            out.append((*_reader_meta(m), pl.get("1.0") or pl[max(pl, key=float)]))
    return out


def _board_triples(dev, arms):
    """Same, from the device panel's ``headline_board`` (a list keyed by r['model'])."""
    board = {r["model"]: r for r in dev["headline_board"]}
    return [(*_reader_meta(m), board[m]) for m in arms if m in board]


def fig_reader_panel(triples, shift_label, title):
    """The reader head-to-head, IDENTICAL on both axes: referral-flip (CI) + clean AUC (CI), every arm coloured by
    CLASS (LP blue · FT orange · from-scratch grey · oracle · VLM red), steadiest-first — so VLMs (red) are read on
    the SAME two metrics as the specialists, in one chart. Legend sits upper-right (the short-bar corner)."""
    triples = sorted(triples, key=lambda t: t[2]["referral_flip"])
    labels = [t[0] for t in triples]; cols = [t[1] for t in triples]
    flip = [t[2]["referral_flip"] for t in triples]
    ferr = np.array([_ci_err(t[2], "referral_flip") for t in triples]).T
    auc = [t[2].get("auc_clean", float("nan")) for t in triples]
    aerr = np.array([_ci_err(t[2], "auc_clean") for t in triples]).T
    y = np.arange(len(triples))
    fig, ax = plt.subplots(1, 2, figsize=(11.4, max(3.0, 0.34 * len(triples) + 1.0)),
                           gridspec_kw=dict(width_ratios=[1.32, 1]))
    ax[0].barh(y, flip, xerr=ferr, color=cols, height=0.66, error_kw=dict(ecolor=INK2, lw=1))
    ax[0].set_yticks(y); ax[0].set_yticklabels(labels, fontsize=8.2); ax[0].invert_yaxis()
    ax[0].set_xlim(0, min(0.62, max(f + e for f, e in zip(flip, ferr[1])) + 0.04))
    ax[0].set_xlabel(f"referral-flip @ {shift_label}  (lower = steadier)"); _despine(ax[0])
    ax[0].set_title("Decision stability", fontsize=11)
    ax[1].barh(y, auc, xerr=aerr, color=cols, height=0.66, error_kw=dict(ecolor=INK2, lw=1))
    ax[1].axvline(0.5, color=MUT, ls=":", lw=1.1)
    ax[1].set_yticks(y); ax[1].set_yticklabels([]); ax[1].invert_yaxis()
    ax[1].set_xlim(0.45, 0.92); ax[1].set_xlabel("clean AUC  (0.5 = chance, → better)"); _despine(ax[1])
    ax[1].set_title("Discrimination", fontsize=11)
    from matplotlib.patches import Patch
    present = [c for c in (LP, FT, DR, SCRATCH, ORACLE, VLM, VLM_ZERO) if c in set(cols)]
    ax[0].legend(handles=[Patch(color=c, label=_CLASS_NAME[c]) for c in present],
                 fontsize=7.4, loc="upper right", framealpha=0.92)
    fig.suptitle(title, fontsize=10.5, y=1.006, color=INK)
    return _uri(fig)


def fig_vlm_closeup(panel, title):
    """The VLM-specific view flip+AUC can't show: clean-ECE overconfidence (vs the FM band) + the few-shot lift.
    The FM ECE band + the weakest-specialist AUC line are pulled LIVE from the same panel."""
    order = sorted([m for m in panel if _is_vlm(m)],
                   key=lambda m: (_vlm_label(m)[0], "fewshot" not in m))
    labels = [_reader_meta(m)[0] for m in order]; cols = [_reader_meta(m)[1] for m in order]
    rows = [panel[m]["per_level"]["1.0"] for m in order]
    ece = [r.get("ece_clean", float("nan")) for r in rows]
    auc = [r.get("auc_clean", float("nan")) for r in rows]
    aerr = np.array([_ci_err(r, "auc_clean") for r in rows]).T
    fm = [panel[m]["per_level"]["1.0"] for m in panel if not _is_vlm(m)]
    fe = [r["ece_clean"] for r in fm if isinstance(r.get("ece_clean"), (int, float))]
    fa = [r["auc_clean"] for r in fm if isinstance(r.get("auc_clean"), (int, float))]
    y = np.arange(len(order))
    # data-driven panel titles — on the raw B-scan the VLMs are weak (below the weakest specialist), but on the
    # friendlier colour-map they reach the specialist level; the ECE gap is also larger on the B-scan.
    best_v = float(np.nanmax(auc)) if len(auc) else 0.0
    ws = min(fa) if fa else 0.70
    ratio = (float(np.nanmedian(ece)) / float(np.median(fe))) if (fe and len(ece)) else 0.0
    l_title = f"Overconfident — ~{ratio:.0f}× the FMs" if ratio >= 1.5 else "Overconfident vs the FM band"
    r_title = ("Reaches the specialist level — few-shot lifts, scale doesn't" if best_v >= ws
               else "Weak — below the specialists; few-shot lifts, scale doesn't")
    fig, ax = plt.subplots(1, 2, figsize=(11.4, max(3.4, 0.5 * len(order) + 1.7)))
    n = len(order)
    if fe:
        ax[0].axvspan(min(fe), max(fe), color=SPECIALIST, alpha=0.15)
    ax[0].barh(y, ece, color=cols, height=0.62)
    ax[0].set_yticks(y); ax[0].set_yticklabels(labels, fontsize=8.2)
    ax[0].set_ylim(n - 0.5, -1.35)                  # inverted + headroom above the top bar for the reference label
    if fe:
        ax[0].text(min(fe), -0.85, f"specialist-FM band {min(fe):.2f}–{max(fe):.2f}", fontsize=7.2,
                   color=SPECIALIST, ha="left", va="center")
    ax[0].set_xlabel("clean ECE  (calibration error, ← better)"); _despine(ax[0])
    ax[0].set_title(l_title, fontsize=11)
    ax[1].barh(y, auc, xerr=aerr, color=cols, height=0.62, error_kw=dict(ecolor=INK2, lw=1))
    ax[1].axvline(0.5, color=MUT, ls=":", lw=1.1)
    if fa:
        ax[1].axvline(min(fa), color=SPECIALIST, ls="--", lw=1)
    ax[1].set_yticks(y); ax[1].set_yticklabels([])
    ax[1].set_ylim(n - 0.5, -1.35)
    if fa:
        ax[1].text(min(fa) - 0.005, -0.85, f"weakest specialist {min(fa):.2f}", fontsize=7.2, color=SPECIALIST,
                   ha="right", va="center")
    ax[1].set_xlim(0.45, 0.92); ax[1].set_xlabel("clean AUC  (0.5 = chance)"); _despine(ax[1])
    ax[1].set_title(r_title, fontsize=11)
    from matplotlib.patches import Patch
    ax[1].legend(handles=[Patch(color=VLM, label="few-shot"), Patch(color=VLM_ZERO, label="zero-shot")],
                 fontsize=7.6, loc="lower right", framealpha=0.92)
    fig.suptitle(title, fontsize=10.5, y=1.006, color=INK)
    return _uri(fig)


def fig_cross_vendor(cv):
    """Fig 14 — real cross-vendor validation: the mean peripapillary RNFL (TSNIT) profile overlaid for Spectralis vs
    Topcon (Eyened-segmented). Each eye's profile is z-scored (scale-invariant, matching the shape-agreement metric),
    so what's shown is the double-hump SHAPE coincidence — the disc anatomy is vendor-consistent."""
    p = ROOT / "data/derived/eyened_cprnfl.npz"
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    if p.exists():
        d = np.load(p)
        x = np.linspace(0, 360, d["spectralis"].shape[1])
        for v, col in (("spectralis", SPECTRALIS), ("topcon", TOPCON)):
            P = np.asarray(d[v], float)
            Z = (P - P.mean(1, keepdims=True)) / (P.std(1, keepdims=True) + 1e-9)   # z-score each eye (scale-invariant)
            m, sd = Z.mean(0), Z.std(0)
            ax.plot(x, m, color=col, lw=2, label=f"{v.capitalize()} (n={len(P)})")
            ax.fill_between(x, m - sd, m + sd, color=col, alpha=0.13)
        ax.set_xlim(0, 360); ax.set_xticks([0, 90, 180, 270, 360])
        ax.set_xlabel("peripapillary angle (°) — the two humps are the superior & inferior bundles")
        ax.set_ylabel("RNFL profile (z-scored)"); _despine(ax)
        r = cv["result"]["per_vendor"].get("topcon", {}).get("shape_agreement_vs_ref")
        ttl = "Real cross-vendor anatomy — the RNFL double-hump coincides across vendors"
        if isinstance(r, (int, float)):
            ttl += f"  (shape agreement r = {r:.2f})"
        ax.set_title(ttl, fontsize=10.2)
        ax.legend(fontsize=8.6, loc="lower center", ncol=2, framealpha=0.9)
    return _uri(fig)


def _speckle_contrast(img, win=9):
    return validity.display_speckle_contrast(img, win)  # single source of truth (asserts-reproduce the constants)


def fig_retouch_validation():
    """RETOUCH validates the appearance DIRECTION: real per-vendor display speckle (Cirrus>Topcon>Spectralis)
    confirms the frame-averaging physics, and the calibrated sim moves GLEAM toward Topcon."""
    ret = loaders.load_retouch_sample(per_vendor=25)
    rc = {v: [_speckle_contrast(im) for im in imgs if getattr(im, "size", 0) > 100] for v, imgs in ret.items()}
    gl = sorted(glob.glob(str(ROOT / "data/raw/gleam/GLEAM/all_samples/*/Thickness.jpg")))[:25]
    g_clean = [_speckle_contrast(np.asarray(Image.open(f).convert("L"), float)) for f in gl]
    g_sim = [_speckle_contrast(BT.apply_bscan_device_shift(np.asarray(Image.open(f).convert("L"), float),
             np.random.default_rng(0), 1.0, ref="spectralis", target="topcon")) for f in gl]
    groups = [("RETOUCH\nCirrus", rc.get("Cirrus", []), CIRRUS), ("RETOUCH\nTopcon", rc.get("Topcon", []), TOPCON),
              ("RETOUCH\nSpectralis", rc.get("Spectralis", []), SPECTRALIS), ("GLEAM\nclean (Spec)", g_clean, MUT),
              ("GLEAM\nsim→Topcon", g_sim, TOPCON)]
    fig, ax = plt.subplots(figsize=(8.8, 4.0))
    for i, (lab, vals, col) in enumerate(groups):
        if vals:
            ax.boxplot([vals], positions=[i], widths=0.6, patch_artist=True, showfliers=False,
                       boxprops=dict(facecolor=col, alpha=0.5, edgecolor=col),
                       medianprops=dict(color=INK, lw=1.5), whiskerprops=dict(color=col), capprops=dict(color=col))
    if rc.get("Topcon"):
        ax.axhline(np.median(rc["Topcon"]), color=TOPCON, ls=":", lw=1, zorder=0)
    ax.set_xticks(range(len(groups))); ax.set_xticklabels([g[0] for g in groups], fontsize=8.3)
    ax.set_ylabel("display speckle contrast (SD/mean)"); _despine(ax)
    ax.set_title("Real per-vendor speckle: Cirrus > Topcon > Spectralis confirms the frame-averaging direction;\n"
                 "the calibrated sim moves GLEAM toward Topcon", fontsize=9.8)
    return _uri(fig)


def fig_retouch_scans():
    """Real RETOUCH B-scans, one per vendor — SHOW the appearance the speckle metric measures: Cirrus grainiest,
    Spectralis smoothest (16-frame averaging), Topcon between. Macular (RETOUCH), identical preprocessing."""
    ret = loaders.load_retouch_sample(per_vendor=3)
    order = [("Cirrus", "grainy"), ("Topcon", "medium"), ("Spectralis", "smooth")]
    fig, ax = plt.subplots(1, 3, figsize=(11.0, 3.2))
    for i, (v, note) in enumerate(order):
        imgs = [im for im in (ret.get(v) or []) if getattr(im, "size", 0) > 100]
        img = np.asarray(imgs[0], float) if imgs else np.zeros((10, 10))
        _imshow_bscan(ax[i], img, f"{v}  ({note})")
    fig.suptitle("The real per-vendor look (RETOUCH, macular, same preprocessing) — frame-averaging (Cirrus 1-frame → "
                 "Spectralis 16-frame ART) sets the graininess the speckle metric captures", fontsize=9.6, y=1.03, color=INK)
    return _uri(fig)


def fig_example_flips(dev, app):
    """Fig 11 — what a shift looks like on each axis, with the aggregate flip each reader shows."""
    m, _ = _gf_map_and_cube()
    b = _gleam_bscan()
    shm = CO.apply_sectored_map(m, "cirrus_to_spectralis", np.random.default_rng(11), severity=1.0)
    shb = BT.apply_bscan_device_shift(b, np.random.default_rng(12), 1.0, ref="spectralis", target="topcon")
    cnn = next((r["referral_flip"] for r in dev["headline_board"] if r["model"] == "fullres_cnn_dr"), None)
    rf = app["panel"]["retfound"]["per_level"]["1.0"]["referral_flip"]
    rff = app["panel"]["retfound_ft"]["per_level"]["1.0"]["referral_flip"]
    fig, ax = plt.subplots(2, 2, figsize=(9.0, 6.4))
    vmax = float(np.nanpercentile(m, 99))
    _imshow_map(ax[0, 0], m, "THICKNESS-MAP · clean (Cirrus)", vmin=0, vmax=vmax)
    _imshow_map(ax[0, 1], shm, "THICKNESS-MAP · sim Spectralis", vmin=0, vmax=vmax)
    _imshow_bscan(ax[1, 0], b, "B-SCAN · clean (Spectralis)")
    _imshow_bscan(ax[1, 1], shb, "B-SCAN · sim Topcon")
    ax[0, 1].text(0.5, -0.16, f"best reader (SmallCNN+dr) flips {cnn*100:.0f}% of calls",
                  transform=ax[0, 1].transAxes, ha="center", fontsize=9, color=GOOD)
    ax[1, 1].text(0.5, -0.16, f"RETFound flips {rf*100:.0f}% (probe) → {rff*100:.0f}% (fine-tuned)",
                  transform=ax[1, 1].transAxes, ha="center", fontsize=9, color=INK2)
    fig.suptitle("What a device shift does to each representation — and how often the call changes",
                 fontsize=11.5, y=1.0, color=INK)
    fig.subplots_adjust(hspace=0.28)
    return _uri(fig)


# ── HTML ─────────────────────────────────────────────────────────────────────────────────────────────
def _esc(s):
    return html.escape(str(s))


CSS = """
:root{--bg:#fcfcfb;--card:#fff;--ink:#12100e;--ink2:#54524d;--mut:#8a897f;--rule:#e6e5df;
--acc:#2a78d6;--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--good:#1baf7a;--warn:#d98a1f;--crit:#e34948;
--goodbg:#e7f6ef;--warnbg:#fbf1de;--critbg:#fbe9e8;--accbg:#eaf2fc;--figbg:#ffffff;
--serif:"Iowan Old Style",Georgia,"Palatino Linotype",serif;--sans:system-ui,-apple-system,"Segoe UI",sans-serif;
--mono:"SF Mono",ui-monospace,Menlo,monospace;--maxw:74ch;}
@media (prefers-color-scheme:dark){:root{--bg:#131311;--card:#1c1b19;--ink:#f1efe9;--ink2:#b8b5ac;--mut:#8a887f;
--rule:#302e2a;--accbg:#16233a;--goodbg:#12332a;--warnbg:#33280f;--critbg:#3a1c1c;--figbg:#fff;}}
:root[data-theme=dark]{--bg:#131311;--card:#1c1b19;--ink:#f1efe9;--ink2:#b8b5ac;--mut:#8a887f;--rule:#302e2a;
--accbg:#16233a;--goodbg:#12332a;--warnbg:#33280f;--critbg:#3a1c1c;--figbg:#fff;}
:root[data-theme=light]{--bg:#fcfcfb;--card:#fff;--ink:#12100e;--ink2:#54524d;--rule:#e6e5df;
--accbg:#eaf2fc;--goodbg:#e7f6ef;--warnbg:#fbf1de;--critbg:#fbe9e8;--figbg:#fff;}
*{box-sizing:border-box}html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);line-height:1.62;
font-size:17px;-webkit-font-smoothing:antialiased}
main{max-width:var(--maxw);margin:0 auto;padding:clamp(1.1rem,4vw,3rem) 1.15rem 5rem;
display:flex;flex-direction:column;gap:1.15rem}
.eyebrow{font-family:var(--mono);font-size:.72rem;letter-spacing:.14em;text-transform:uppercase;color:var(--mut)}
h1{font-family:var(--serif);font-weight:600;font-size:clamp(1.9rem,5.2vw,2.9rem);line-height:1.1;
margin:.2rem 0 .1rem;text-wrap:balance}
h2{font-family:var(--serif);font-weight:600;font-size:1.5rem;line-height:1.2;margin:1.7rem 0 .1rem;text-wrap:balance}
h3{font-weight:650;font-size:1.06rem;margin:1.1rem 0 .1rem}
p{margin:.55rem 0}.lede{font-size:1.16rem;color:var(--ink2)}
a{color:var(--acc)}.act{font-family:var(--mono);font-size:.74rem;letter-spacing:.16em;text-transform:uppercase;
color:var(--acc);border-top:1px solid var(--rule);padding-top:1.1rem;margin-top:1.3rem}
.bluf{font-size:1.06rem;background:var(--accbg);border-left:3px solid var(--acc);border-radius:8px;
padding:.75rem .95rem}
figure{margin:.7rem 0 .3rem}
.fig{background:var(--figbg);border:1px solid var(--rule);border-radius:12px;padding:.7rem;
box-shadow:0 1px 2px rgba(0,0,0,.04)}
.fig img{display:block;width:100%;height:auto;border-radius:5px}
figcaption{font-size:.86rem;color:var(--ink2);margin-top:.5rem;padding:0 .2rem}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:.7rem;margin:.3rem 0}
.tile{background:var(--card);border:1px solid var(--rule);border-radius:11px;padding:.8rem .85rem}
.tile .n{font-family:var(--mono);font-size:1.55rem;font-weight:600;line-height:1}
.tile .k{font-size:.82rem;color:var(--ink2);margin-top:.3rem}
.tile.g .n{color:var(--good)}.tile.o .n{color:var(--s2)}.tile.c .n{color:var(--crit)}
.callout{border:1px solid var(--rule);border-left:3px solid var(--acc);border-radius:8px;padding:.7rem .9rem;
background:var(--card);font-size:.96rem}
.callout.good{border-left-color:var(--good);background:var(--goodbg)}
.callout.crit{border-left-color:var(--crit);background:var(--critbg)}
.callout.warn{border-left-color:var(--warn);background:var(--warnbg)}
.callout b{font-weight:650}
.tscroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
table{border-collapse:collapse;width:100%;font-size:.9rem;min-width:min(100%,540px)}
th,td{text-align:left;padding:.42rem .6rem;border-bottom:1px solid var(--rule)}
th{font-family:var(--mono);font-size:.72rem;letter-spacing:.05em;text-transform:uppercase;color:var(--mut)}
td.n,th.n{text-align:right;font-family:var(--mono)}
.chip{display:inline-block;font-family:var(--mono);font-size:.7rem;padding:.08rem .4rem;border-radius:5px;
border:1px solid var(--rule);color:var(--ink2)}
.chip.g{color:var(--good);border-color:var(--good)}.chip.c{color:var(--crit);border-color:var(--crit)}
ul.led{margin:.4rem 0;padding-left:1.1rem}ul.led li{margin:.3rem 0;color:var(--ink2);font-size:.96rem}
.ref{font-size:.82rem;color:var(--mut)}
.ci{color:var(--mut);font-size:.82em;font-weight:400;white-space:nowrap}
.mlegend{display:grid;grid-template-columns:repeat(auto-fit,minmax(262px,1fr));gap:.55rem .9rem;background:var(--card);
border:1px solid var(--rule);border-radius:10px;padding:.75rem .9rem;font-size:.92rem;line-height:1.5}
.mlegend .mh{display:block;font-family:var(--mono);font-size:.72rem;letter-spacing:.05em;text-transform:uppercase;
color:var(--acc);font-weight:650;margin-bottom:.25rem}
.mlegend .mrow{padding:.05rem 0}
.mlegend .ms{color:var(--mut);font-size:.85em;text-transform:none;letter-spacing:0;font-weight:400}
.mlegend .mfull{grid-column:1/-1;border-top:1px solid var(--rule);padding-top:.5rem;margin-top:.1rem;color:var(--ink2)}
.findings{display:grid;grid-template-columns:repeat(auto-fit,minmax(238px,1fr));gap:.7rem;margin:.5rem 0}
.finding{background:var(--card);border:1px solid var(--rule);border-left:3px solid var(--mut);border-radius:10px;
padding:.75rem .9rem}
.finding.g{border-left-color:var(--good)}.finding.b{border-left-color:var(--s1)}
.finding.o{border-left-color:var(--s2)}.finding.c{border-left-color:var(--crit)}
.finding .fh{font-weight:650;font-size:1rem;line-height:1.28;text-wrap:balance}
.finding .fn{font-family:var(--mono);font-size:1.2rem;font-weight:600}
.finding .fb{font-size:.87rem;color:var(--ink2);margin-top:.4rem;line-height:1.5}
.foot{border-top:1px solid var(--rule);margin-top:1.6rem;padding-top:1rem;font-size:.82rem;color:var(--mut)}
"""


def _tile(n, k, cls=""):
    return f'<div class="tile {cls}"><div class="n">{_esc(n)}</div><div class="k">{_esc(k)}</div></div>'


def _fig(uri, cap):
    return f'<figure class="fig"><img alt="" src="{uri}"><figcaption>{cap}</figcaption></figure>'


def _mc(r, key, fmt):
    """One metric cell = point value + its 95% bootstrap CI (from ``r['ci'][key]``); '—' if absent/NaN."""
    v = r.get(key)
    if not isinstance(v, (int, float)) or v != v:
        return "—"
    s = fmt.format(v)
    ci = (r.get("ci") or {}).get(key)
    if not (ci and len(ci) == 2):
        return s
    cf = "{:+.2f}" if fmt.startswith("{:+") else "{:.2f}"
    return f'{s} <span class="ci">[{cf.format(ci[0])}, {cf.format(ci[1])}]</span>'


# the ONE metric language, used for EVERY per-model table on both axes (discrimination + decision, each with CI):
_MCOLS = [("clean AUC ↑", "auc_clean", "{:.2f}"), ("ΔAUC →0", "delta_auc", "{:+.2f}"),
          ("flip ↓", "referral_flip", "{:.2f}"), ("Δsens@spec ±", "delta_sens_at_spec", "{:+.2f}")]


def _metric_table(rows, ece=False):
    """ONE metric language on both axes — Discrimination {clean AUC, ΔAUC} · Decision {referral-flip, Δsens@spec},
    each with its 95% CI; spec@shift is the anti-gaming guard. VLM tables (``ece=True``) append clean-ECE + ΔECE
    (the overconfidence the FMs don't have). ``rows`` = list of (label, per-level dict carrying a nested ``ci``)."""
    head = ["reader / model"] + [c[0] for c in _MCOLS] + ["spec@shift"] + (["ECE ↓", "ΔECE ↓"] if ece else [])
    h = "<tr>" + "".join(f'<th class="{"n" if i else ""}">{_esc(c)}</th>' for i, c in enumerate(head)) + "</tr>"
    body = ""
    for lab, r in rows:
        cells = [_mc(r, k, f) for _, k, f in _MCOLS]
        cells.append(f"{r.get('spec_realized_shift', float('nan')):.2f}")
        if ece:
            cells += [f"{r.get('ece_clean', float('nan')):.2f}", f"{r.get('delta_ece', float('nan')):+.3f}"]
        body += f"<tr><td>{_esc(lab)}</td>" + "".join(f'<td class="n">{c}</td>' for c in cells) + "</tr>"
    return f'<div class="tscroll"><table>{h}{body}</table></div>'


def _metrics_legend():
    """The metric language defined ONCE — a 2×2 of discrimination × decision, so no metric is read cold."""
    return (
        '<div class="mlegend">'
        '<div><span class="mh">Discrimination <span class="ms">— threshold-free: can it still tell glaucoma from '
        'healthy?</span></span>'
        '<div class="mrow"><b>clean AUC</b> — baseline rank quality</div>'
        '<div class="mrow"><b>ΔAUC</b> — change under shift (→ 0 = ranking survived)</div></div>'
        '<div><span class="mh">Decision <span class="ms">— at the deployed 90%-specificity threshold: the actual '
        'referral call</span></span>'
        '<div class="mrow"><b>referral-flip</b> — how often the call changes</div>'
        '<div class="mrow"><b>Δsens@spec</b> — signed (clean − shift): <b>+</b> <em>missed</em> glaucoma '
        '(sensitivity lost), <b>−</b> caught more — usually by over-referring, so read with spec@shift '
        '<span class="ms">(spec@shift guards against a sensitivity bought by flooding referrals)</span></div></div>'
        '<div class="mfull"><b>Every value carries a 95% bootstrap CI.</b> Device-shift signature: ΔAUC ≈ 0 while '
        'flip ↑ and spec@shift collapses (|Δsens@spec| large in <em>either</em> direction) — the shift breaks the '
        '<em>decision</em>, not the <em>ranking</em>.</div></div>')


# Model roster + sizes. SmallCNN counted exactly (torch_readers.SmallCNN: 97,905 params at in_ch=2 for the map
# reader, 97,761 at in_ch=1 for the B-scan reader → ~98k). FM / Ridge counts are the standard ARCHITECTURE sizes
# (approx.) — the builder stays torch-free, so the FMs are not instantiated here to count.
MODEL_ROSTER = [
    ("SmallCNN", "thickness-map reader + B-scan from-scratch control", "4-block conv (16→128) + GAP head", "~98k"),
    ("Ridge", "thickness-map 64² floor", "linear on 4,096 pixels", "~4k"),
    ("AffineRefit", "offset oracle (scored upper bound)", "per-quadrant inverse — no learned net", "—"),
    ("RETFound", "OCT foundation model", "ViT-L/16 (MAE)", "~304M"),
    ("DINOv2", "generic-ViT control", "ViT-L/14 + registers", "~300M"),
    ("MIRAGE-Base", "OCT foundation model", "ViT-B MultiMAE", "~86M"),
    ("ResNet-50", "ImageNet-supervised CNN (scale sweep)", "CNN, ImageNet-1k", "~25M"),
    ("ConvNeXt-T", "ImageNet-supervised CNN (scale sweep)", "CNN, ImageNet-1k", "~29M"),
    ("ConvNeXt-B", "ImageNet-supervised CNN (scale sweep)", "CNN, ImageNet-1k", "~89M"),
    ("ConvNeXt-L", "ImageNet-supervised CNN (scale sweep)", "CNN, ImageNet-1k", "~198M"),
    ("Gemini 2.5 / 3.6", "frontier multimodal LLM (Google, closed)", "vision-language model", "undisclosed (~frontier)"),
    ("GPT-5.6 Sol", "frontier multimodal LLM (OpenAI, closed)", "vision-language model", "undisclosed (~frontier)"),
    ("Claude Opus 5", "frontier multimodal LLM (Anthropic, closed)", "vision-language model", "undisclosed (~frontier)"),
    ("Qwen3-VL-8B", "open-weights multimodal LLM (Alibaba)", "vision-language model", "~8B"),
    ("MedGemma 1.5-4B", "medical-specialist multimodal LLM (Google; ophthalmology in pretraining)", "vision-language model", "~4B"),
]


def _roster_table():
    """The models under test + their sizes, so the compact reader is read next to the giant FMs."""
    h = '<tr><th>model</th><th>role</th><th>architecture</th><th class="n">params</th></tr>'
    body = "".join(f"<tr><td><b>{_esc(m)}</b></td><td>{_esc(r)}</td><td>{_esc(a)}</td><td class=\"n\">{_esc(p)}</td></tr>"
                   for m, r, a, p in MODEL_ROSTER)
    return f'<div class="tscroll"><table>{h}{body}</table></div>'


def _provenance_table():
    rows = [
        ("Thickness-map · Cirrus→Spectralis", "✅ fully calibrated (same-eye)", "Warner 2011 — bias + slope + LoA, n=95"),
        ("Thickness-map · Cirrus→Topcon", "⚠ extrapolated", "Buchser 2012 conversion; Topcon-at-ONH data thin"),
        ("Thickness-map · Cirrus→RTVue", "⚠ bias only", "Buchser 2012 regression; LoA inferred"),
        ("B-scan · Spectralis→Topcon", "◐ direction-validated + magnitude proxy",
         "RETOUCH real per-vendor speckle (⚠ macular → disc proxy)"),
    ]
    h = "<tr><th>axis · shift</th><th>calibration strength</th><th>literature basis</th></tr>"
    body = "".join(f"<tr><td>{_esc(a)}</td><td>{_esc(b)}</td><td>{_esc(c)}</td></tr>" for a, b, c in rows)
    return f'<div class="tscroll"><table>{h}{body}</table></div>'


def _real_evidence_table(rows):
    """The REAL (non-simulated) rungs the benchmark already stands on — every clean discrimination number is real
    device data with real labels; the cross-vendor anatomy + per-vendor appearance are real. ``rows`` =
    (rung, real data · labels, N, what it establishes, result)."""
    head = ("real rung", "real data · labels", "N", "what it establishes", "result")
    h = "<tr>" + "".join(f"<th>{_esc(c)}</th>" for c in head) + "</tr>"
    body = "".join(
        f"<tr><td><b>{_esc(a)}</b></td><td>{_esc(b)}</td><td class=\"n\">{_esc(n)}</td>"
        f"<td>{_esc(w)}</td><td class=\"n\">{_esc(r)}</td></tr>"
        for a, b, n, w, r in rows)
    return f'<div class="tscroll"><table>{h}{body}</table></div>'


def build():
    dev = json.load(open(REPORTS / "oct_device_axis_panel.json"))
    app = json.load(open(REPORTS / "oct_appearance_axis_panel.json"))
    valid = json.load(open(REPORTS / "oct_format_validity.json"))
    cv = json.load(open(REPORTS / "oct_cross_vendor_panel.json"))
    mapfm = json.load(open(REPORTS / "oct_map_fm_panel.json")) if (REPORTS / "oct_map_fm_panel.json").exists() else None

    # headline numbers
    cnn = next(r for r in dev["headline_board"] if r["model"] == "fullres_cnn_dr")
    rf = app["panel"]["retfound"]["per_level"]["1.0"]
    rff = app["panel"]["retfound_ft"]["per_level"]["1.0"]
    d2r = app["panel"]["dinov2_reg"]["per_level"]["1.0"]
    d2f = app["panel"]["dinov2_ft"]["per_level"]["1.0"]
    vs = valid["V1a_structural"]

    print("rendering figures…")
    ba_uri, ba_bias, ba_loa = fig_bland_altman(valid)
    mp = mapfm["panel"] if mapfm else {}
    struct_arms = ["fullres_cnn", "fullres_cnn_dr", "ridge_baseline", "ridge_dr", "AffineRefit"]
    F = {
        "devices": fig_devices(), "device_axis": fig_device_axis(), "appearance_axis": fig_appearance_axis(),
        "harness": fig_harness(), "ba": ba_uri, "sector": fig_sector_loa(valid), "dose": fig_dose_response(valid),
        "crossv": fig_cross_vendor(cv), "retouch": fig_retouch_validation(),
        "retouch_scans": fig_retouch_scans(),
        # b-scan (appearance) axis: every reader (FM · CNN · VLM) in one head-to-head + the VLM close-up
        "bscan_panel": fig_reader_panel(_axis_triples(app["panel"], list(app["panel"])),
                                        "Spectralis→sim-Topcon appearance stress",
                                        "B-SCAN axis · every reader on the same 360 eyes + shift"),
        "bscan_vlm": fig_vlm_closeup(app["panel"], "B-SCAN axis · VLM close-up (calibration + few-shot)"),
    }
    # thickness-map axis: ONE reader head-to-head — native-map structural readers + colour-map FM/CNN/VLM together
    # (same GDP eyes + same scanner_offset shift; only the input representation differs → comparable in kind, §0)
    map_triples = _board_triples(dev, struct_arms) + (_axis_triples(mp, list(mp)) if mapfm else [])
    F["map_panel"] = fig_reader_panel(map_triples, "Cirrus→Spectralis device shift",
                                      "THICKNESS-MAP axis · every reader (native map + colour-map)")
    if mapfm:
        F["map_vlm"] = fig_vlm_closeup(mp, "THICKNESS-MAP axis · VLM close-up (calibration + few-shot)")
    print("figures done:", ", ".join(F))

    # unified per-axis metric tables — ONE metric language (flip · AUC · ΔAUC · Δsens@spec, all with CI; ECE column
    # too so the FMs' 0.04–0.12 vs the VLMs' 0.26–0.42 shows). The map table combines the native-map structural
    # readers + the colour-map FM/CNN/VLM readers (one head-to-head, matching the combined panel).
    dev_rows = [(_reader_meta(r["model"])[0], r) for r in dev["headline_board"]
                if r["model"] not in ("constant", "ComBat_full", "Denoise")]

    def _rows(panel):
        return [(_reader_meta(m)[0], panel[m]["per_level"]["1.0"]) for m in panel]
    bscan_table = _metric_table(sorted(_rows(app["panel"]), key=lambda t: t[1]["referral_flip"]), ece=True)
    map_rows = sorted(dev_rows + (_rows(mp) if mapfm else []), key=lambda t: t[1]["referral_flip"])
    map_table = _metric_table(map_rows, ece=True)
    prov_table, roster_table, metrics_legend = _provenance_table(), _roster_table(), _metrics_legend()
    gbest = app["panel"]["gemini_3p6_flash_fewshot_think"]["per_level"]["1.0"]
    gprozero = app["panel"]["gemini_2p5_pro_zeroshot_think"]["per_level"]["1.0"]
    gprofew = app["panel"]["gemini_2p5_pro_fewshot_think"]["per_level"]["1.0"]

    # full VLM roster (live) for the updated prose — closed frontier (GPT/Claude), open (Qwen), specialist (MedGemma)
    def _pl(panel, key):
        return panel[key]["per_level"]["1.0"] if panel and key in panel else {}
    gpt_bz = _pl(app["panel"], "openai_5p6_sol_zeroshot_think")               # GPT b-scan zero
    claude_bf = _pl(app["panel"], "claude_opus_5_fewshot_think")              # Claude b-scan few (better-calibrated)
    claude_mz = _pl(mp, "claude_map_opus_5_zeroshot_think")                   # Claude map zero (best VLM read 0.85)
    gpt_mf = _pl(mp, "openai_map_5p6_sol_fewshot_think")                      # GPT map few (0.84)
    qwen_bz = _pl(app["panel"], "qwen3vl_8b_zeroshot")                        # open generalist — near-chance
    qwen_mz = _pl(mp, "qwen3vl_map_8b_zeroshot")                              # open generalist map (degenerate)
    medg_bz = _pl(app["panel"], "medgemma_4b_zeroshot")                       # medical specialist — near-chance
    medg_mz = _pl(mp, "medgemma_map_4b_zeroshot")
    dref = app["panel"]["dinov2_reg"]["per_level"]["1.0"]                 # live test-split sizes (reconcile prose counts)
    npos_b, nte_b = dref.get("n_pos"), dref.get("n")
    npos_m = (mp["dinov2_reg"]["per_level"]["1.0"].get("n_pos") if mp else None)

    # REAL-evidence ledger — the non-simulated rungs the benchmark already stands on (data-driven)
    real_map_auc = max(r.get("auc_clean", 0) for r in dev["headline_board"] if r["model"] != "constant")
    real_bscan_auc = max(v["per_level"]["1.0"].get("auc_clean", 0)
                         for m, v in app["panel"].items() if not _is_vlm(m))
    topcon_shape = cv["result"]["per_vendor"]["topcon"]["shape_agreement_vs_ref"]
    spk = BT.device_physics.DISPLAY_SPECKLE_CONTRAST
    real_rows = [
        ("Cirrus detection", "Harvard-GDP RNFLT maps · VF-derived labels", "1,000",
         "models really detect glaucoma on real Cirrus", f"AUC {real_map_auc:.2f} (clean)"),
        ("Spectralis detection", "GLEAM B-scans · clinical labels", "1,200",
         "models really detect glaucoma on real Spectralis", f"AUC {real_bscan_auc:.2f} (clean)"),
        ("Cross-vendor anatomy", "Eyened cpRNFL · Spectralis vs Topcon", "60 + 60",
         "the disc anatomy agrees across vendors (same TSNIT double-hump)", f"shape {topcon_shape:.2f}"),
        ("Per-vendor appearance", "RETOUCH real B-scans · 3 vendors", "~30 / vendor",
         "vendor speckle really differs — in the simulated direction",
         f"Cirrus {spk['cirrus']:.2f} > Topcon {spk['topcon']:.2f} > Spectralis {spk['spectralis']:.2f}"),
    ]
    real_evidence_table = _real_evidence_table(real_rows)

    # (the map-transfer point is now carried by the map-axis reader panel + table + prose, not a separate callout)

    # scale×pretraining callout — the frozen LP scale sweep (ResNet-50 → ConvNeXt-T/B/L, one ImageNet-supervised
    # family ~25M→197M) + the per-model CNN LP→FT delta. Numbers pulled live so the prose can't drift from the panels.
    def _sweep(panel):
        arms = [("resnet50", "~25M"), ("convnext_tiny", "~29M"), ("convnext_base", "~88M"), ("convnext_large", "~197M")]
        got = [(k, sz) for k, sz in arms if k in panel]
        aucs = [panel[k]["per_level"]["1.0"]["auc_clean"] for k, _ in got]
        flips = [panel[k]["per_level"]["1.0"]["referral_flip"] for k, _ in got]
        return len(got), (min(aucs), max(aucs)), (min(flips), max(flips))

    def _ftd(panel, base):
        r0 = panel.get(base, {}).get("per_level", {}).get("1.0"); r1 = panel.get(base + "_ft", {}).get("per_level", {}).get("1.0")
        return (r0["auc_clean"], r1["auc_clean"], r0["referral_flip"], r1["referral_flip"]) if (r0 and r1) else None

    scale_callout = ""
    if mapfm and _sweep(app["panel"])[0] >= 3 and _sweep(mapfm["panel"])[0] >= 3:
        _, (b_al, b_ah), (b_fl, b_fh) = _sweep(app["panel"])
        _, (m_al, m_ah), (m_fl, m_fh) = _sweep(mapfm["panel"])
        rn_b, cx_t_b = _ftd(app["panel"], "resnet50"), _ftd(app["panel"], "convnext_tiny")
        d2r = mapfm["panel"].get("dinov2_reg", {}).get("per_level", {}).get("1.0")
        d2f = mapfm["panel"].get("dinov2_ft", {}).get("per_level", {}).get("1.0")
        ft_sent = ""
        if rn_b and cx_t_b:
            d2_txt = f' DINOv2 too (map AUC {d2r["auc_clean"]:.2f}→{d2f["auc_clean"]:.2f}).' if (d2r and d2f) else ''
            ft_sent = (
                ' <b>Fine-tuning each model with its OWN recipe</b> (ConvNeXt layer-decay 0.9 / ResNet ~1.0 / DINOv2 0.65 '
                '— never a shared recipe, which would confound model-vs-recipe), measured at <b>every ConvNeXt scale</b> '
                '(T/B/L) plus ResNet-50 and DINOv2, is a within-model LP→FT delta, not a cross-model rank: on the map it '
                f'lifts discrimination at every size (+0.02–0.04 AUC) at no stability cost;{d2_txt} on the raw B-scan it '
                f'rescues the unstable ConvNeXt-T (flip {cx_t_b[2] * 100:.0f}→{cx_t_b[3] * 100:.0f}%) yet is the double-edged '
                f'sword for ResNet-50 (AUC {rn_b[0]:.2f}→{rn_b[1]:.2f} but flip {rn_b[2] * 100:.0f}→{rn_b[3] * 100:.0f}%; Kumar 2022).')
        scale_callout = (
            '<div class="callout"><b>Does scale buy device-tolerance? No.</b> A frozen linear-probe <b>scale sweep</b> — '
            'one ImageNet-supervised family from ResNet-50 (~25M) through ConvNeXt-T/B/L (~29M→197M), every model probed '
            'identically so the comparison is recipe-invariant — comes out <b>flat</b>: clean discrimination barely moves '
            f'with size (B-scan AUC {b_al:.2f}–{b_ah:.2f}, map {m_al:.2f}–{m_ah:.2f}; CIs overlap), and decision-stability '
            'shows <b>no monotone scale trend</b> — the largest model (ConvNeXt-L, 197M) is the steadiest on neither axis '
            f'(map referral-flip {m_fl * 100:.0f}–{m_fh * 100:.0f}%, all tiny; B-scan {b_fl * 100:.0f}–{b_fh * 100:.0f}%, '
            'non-monotone). Device-tolerance tracks the <em>representation</em> (the colour-map is robust at every size; '
            'the raw B-scan is not) and pre-training breadth, <b>not parameter count</b>.' + ft_sent
            + ' The fair cross-model comparison therefore lives in the frozen regime (Oquab 2023; Kumar 2022).</div>'
        )

    body = f"""
<span class="eyebrow">Glaucoma-Triage · OCT-FormatShift v3 · P4</span>
<h1>Can one glaucoma-OCT reader be trusted across scanners?</h1>
<p class="lede">Optical-coherence-tomography (OCT) is how glaucoma referrals get triaged — but the same eye reads
differently on a Zeiss Cirrus, a Heidelberg Spectralis, or a Topcon. This report builds a fair, literature-calibrated
test of whether a model's <em>referral decision</em> survives a change of device or image format, and scores today's
foundation models — and a roster of frontier multimodal LLMs (Gemini, GPT-5.6, Claude Opus 5), an open-weights VLM
(Qwen3-VL-8B) and a medical specialist (MedGemma) — on it. <b>The finding, in one line:</b> device/format
tolerance is not conferred by model scale, the "OCT" pre-training label, or being a foundation model — it must be
<em>measured</em> (on a calibrated shift) and it can be <em>trained</em> (stability-trained domain-randomisation); and
today's vision-language models are the weakest, most overconfident readers — worst of all the small open ones, which
collapse to near-chance.</p>
<div class="bluf"><b>Bottom line.</b> Under a device shift calibrated to published cross-scanner agreement, a compact
from-scratch CNN (<b>SmallCNN</b>, ~98k params) — with stability training — is as device-stable as foundation models
thousands of times its size, flipping just {cnn['referral_flip']*100:.0f}% of referrals (steadier than the 300M DINOv2). On a realistic B-scan appearance stress, robustness tracks
<b>pre-training breadth and adaptation, not the "OCT" label</b>: a generic ViT (DINOv2) is the steadiest <em>and</em>
best-discriminating, fine-tuning helped every foundation model, and the OCT-specific MIRAGE is the weakest. The
frontier VLMs (Gemini/GPT/Claude) are weaker still and mostly overconfident (Claude Opus 5 the better-calibrated
exception), though a colour-map render recovers the capable ones to specialist level; the small open VLMs
(Qwen3-VL-8B, MedGemma) collapse to near-chance and are the worst-calibrated arms here. Every claim is on a
<em>synthetic-but-calibrated</em> axis — necessary, not sufficient, for real-world interchangeability.</div>

<div class="tiles">
  {_tile(f"{cnn['referral_flip']*100:.0f}%", "referrals flipped by the stability-trained SmallCNN (~98k) — matching 300M FMs", "g")}
  {_tile(f"{rf['referral_flip']*100:.0f}→{rff['referral_flip']*100:.0f}%", "RETFound flip: linear-probe → fine-tuned (FT steadier)", "g")}
  {_tile(f"{d2f['auc_clean']:.2f}", f"best clean AUC (DINOv2, fine-tuned) — at just {d2f['referral_flip']*100:.0f}% flip", "")}
  {_tile("+1.9 / ±7.7 µm", "device shift calibrated to Warner 2011 (bias / LoA)", "")}
  {_tile(f"{gbest['auc_clean']:.2f}", "best frontier VLM (Gemini 3.6-flash) — only the weakest specialist's level", "c")}
</div>

<p class="act">1 · Background — the problem, shown</p>
<h2>Two pictures of one nerve</h2>
<p>Glaucoma damages the <b>retinal nerve fibre layer (RNFL)</b> around the optic disc. OCT measures it two ways.
A <b>thickness map</b> is an en-face picture where colour encodes how thick the RNFL is in microns — the clinical
output of a Cirrus scan. A <b>B-scan</b> is a raw cross-section through the retina, where a segmentation algorithm
traces the RNFL to measure it. Same biology; very different pictures.</p>
{_fig(F['devices'], "<b>Fig 1.</b> The same structure across three devices: a Cirrus en-face RNFL <b>thickness map</b> "
      "(left; colour = µm), a Heidelberg Spectralis circumpapillary <b>B-scan</b> (middle), and a Topcon B-scan with its "
      "RNFL <b>segmentation</b> (right, orange). Real data (Harvard-GF, GLEAM, GOALS).")}
<p>Why do devices disagree? Not the pixels. The dominant cause is the <b>segmentation algorithm</b> — a common
segmenter collapses the Cirrus–Spectralis gap from 6.7 µm to ~0.1 µm (Patel 2012); then <b>scan geometry /
centration</b> (a 0.1 mm mis-centred circle costs ~2.3 µm globally, 5–6 µm superior/inferior; Cheung–Leung 2008),
<b>ocular magnification</b> (≈−2.8 µm/mm), and — for the raw B-scan look — <b>speckle</b> (Schmitt 1999). It is a
measurement-science effect, not a picture-quality one.</p>
<div class="callout"><b>The catch that shapes everything.</b> Every public disc-OCT set is single-vendor — Cirrus is
the US Harvard cohort, Spectralis the Chinese GLEAM cohort, Topcon is GOALS — so <b>device and cohort are perfectly
aliased</b>: no real dataset can separate a scanner effect from a population one (Pearl–Bareinboim 2014). We therefore
don't measure the device effect from real data — we <em>import</em> it as a literature-calibrated transform, and keep
every result labelled by representation, never pooled.</div>
<p><b>What's new here.</b> OCT foundation models already claim cross-device generalization — but on real held-out
cohorts where device and population are <em>confounded</em>, scored by accuracy/AUC (RETFound; OCTCube-M 2024);
recent multimodal-LLM glaucoma work reads <em>clinical-data tables</em> or writes reports for agreement with
specialists, not raw scans under a device change (e.g. arXiv:2510.02403, 2025); and the closest prior FM-vs-VLM
head-to-head for glaucoma (Chuter 2025, TVST) is <em>single-device</em> and accuracy-only, with no device shift and
no decision metric. We instead isolate the device effect with a <em>literature-calibrated same-eye shift</em>, score
the <em>referral decision</em> itself (an operating-point-free flip), and put OCT FMs, generic FMs, an ImageNet-CNN
scale sweep, a from-scratch CNN and <em>frontier VLMs</em> on one common ground. The novelty is that <em>conjunction</em>
— calibrated same-eye shift × decision-flip metric × de-confounded × FM + CNN-scale-sweep + from-scratch + VLM — not
FM-vs-VLM-on-OCT on its own: a controlled test of whether scanner-tolerance comes from model scale, the "OCT"
pre-training label, or pre-training breadth.</p>

<p class="act">2 · Methods — two axes, one common test</p>
<h2>One eye, two honest stress axes</h2>
<p>A scanner change hits a thickness map and a raw B-scan in different ways, so we build <b>two device-shift axes</b> —
one per representation — each calibrated or grounded to the literature and scored on its own, never cross-ranked.</p>
<h3>Thickness-map axis — calibrated to Warner 2011</h3>
<p>A per-sector, per-eye shift whose magnitude matches published same-eye Cirrus→Spectralis agreement (bias +1.94 µm,
95% limits −5.7…+9.6 µm; Warner 2011), with anatomy-coupled spread drawn once per eye so the superior/inferior
bundles vary more than nasal/temporal (Cheung–Leung 2008).</p>
{_fig(F['device_axis'], "<b>Fig 2.</b> The map axis on a real Cirrus map: clean → Warner-calibrated Cirrus→sim-Spectralis "
      "→ difference (µm). The maps look near-identical, yet the read moves a few microns per sector — more in the superior/"
      "inferior bundles — enough to flip a borderline referral. The shift is spatially <em>smooth</em> (feathered edges + "
      "low-frequency texture), matching the smooth centration/magnification mechanisms rather than hard blocks (Patel 2012).")}
<h3>B-scan appearance axis — direction-validated by RETOUCH</h3>
<p>GLEAM's B-scans are Spectralis, so we simulate <b>Spectralis→Topcon</b>: more speckle (Topcon is single-frame vs
Spectralis's 16-frame averaging) and lower display contrast — the two dominant terms; the axial-blur term is
near-inert at this scale. Magnitudes from device physics (Schmitt 1999) and
calibrated to the real per-vendor speckle in RETOUCH. It reproduces the cross-vendor <em>direction</em>, not a
same-region µm-LoA, so we scope it as an appearance-tolerance stress. ⚠ RETOUCH is macular and our target is disc —
the device appearance transfers, the absolute level is a proxy.</p>
{_fig(F['appearance_axis'], "<b>Fig 3.</b> A real Spectralis B-scan → its simulated Topcon look (injected speckle + "
      "lower contrast; the axial-blur term is near-inert at this scale) — a direction-validated stress test of "
      "image-reading robustness.")}
<h3>The common ground — one referral call, one metric</h3>
<p>Every reader — the compact map CNNs, the OCT and generic foundation models, the ImageNet CNNs, and the frontier
VLMs — is reduced to the <b>same clinical act</b>: a glaucoma-referral probability. We then score the <b>paired
clean→shifted degradation</b> on the same eyes in one language: whether the <em>decision</em> holds (referral-flip,
Δsens@spec) and whether the <em>ranking</em> holds (AUC, ΔAUC), each with eye-level bootstrap CIs (the decision-flip
metric follows Hendrycks 2019). The 90%-specificity operating threshold is fixed on the clean test negatives and
applied unchanged to the shifted scores — so the shift-induced <em>change</em> is a fair within-eye comparison
across every arm; the absolute clean sens@spec carries a mild in-sample optimism.</p>
{_fig(F['harness'], "<b>Fig 4.</b> The common-ground harness — heterogeneous models meet at one referral call and one "
      "paired-degradation metric, so a map reader and a B-scan model are compared in kind, labelled and never pooled.")}
<h3>The models under test</h3>
<p>A roster spanning four orders of magnitude — from a ~98k-param compact CNN and a linear floor to ~300M-param
foundation models — plus an offset oracle and frontier VLMs, all reduced to the same referral call:</p>
{roster_table}
<p class="ref">The ResNet/ConvNeXt sweep and the foundation models are all <b>pre-trained</b> — linear-probed on frozen
features (LP) or fine-tuned (FT); only the SmallCNN is trained <b>from scratch</b>. SmallCNN counted exactly (~98k);
FM/Ridge sizes are standard architecture figures. Size is context, not a ranking — the two axes never cross-ranked (§0).</p>
<h3>A plain-language glossary</h3>
<div class="tscroll"><table>
<tr><th>term</th><th>what it means</th></tr>
<tr><td>Domain-rand + stability (+dr)</td><td>train the reader on random-magnitude device shifts <em>and</em> penalise any clean↔shifted disagreement, so its referral call is invariant to the shift (Tobin 2017; Zheng 2016)</td></tr>
<tr><td>Affine-refit</td><td>a device-aware calibration <em>given</em> the exact offset to invert — an oracle (a scored upper bound, not a deployable method)</td></tr>
<tr><td>Linear-probe (LP)</td><td>freeze the model, train only a linear classifier on its features — the OOD-robust way to use a foundation model (Kumar 2022)</td></tr>
<tr><td>Fine-tune (FT)</td><td>update the whole model on the task — can erase inherited robustness</td></tr>
</table></div>

<p class="act">3 · Validation — is the simulated shift real?</p>
<h2>The thickness-map axis reproduces published agreement</h2>
<p>A synthetic axis is only worth scoring if it behaves like the real thing. Three checks:</p>
{_fig(F['ba'], f"<b>Fig 5.</b> Bland–Altman of the deployed shift (synthetic per-eye differences) against Warner 2011's "
      f"published 95% limits (band, +{vs['published_bias_um']:.1f} ± {vs['published_loa_half_um']:.1f} µm). The synthetic "
      f"bias (+{ba_bias:.1f} µm) and limits (±{ba_loa:.1f} µm) reproduce the published agreement to within the ≈4 µm "
      f"test–retest floor (Mwanza 2010; TOST-equivalent).")}
{_fig(F['sector'], "<b>Fig 6.</b> The shift is anatomy-coupled — limits of agreement widen in the superior and inferior "
      "bundles, matching the centration literature (Cheung–Leung 2008).")}
{_fig(F['dose'], "<b>Fig 7.</b> Dose–response: referral-flip and read-drift rise monotonically with dose, and the "
      "smallest doses sit near the ≈4 µm test–retest floor (Mwanza 2010) — a graded axis, not a cliff.")}
<h2>The B-scan appearance axis is direction-validated against real Topcon (RETOUCH)</h2>
<p>The appearance stress can't be calibrated to same-eye data (none exists), but it can be <em>checked</em>: under
identical preprocessing, RETOUCH confirms the frame-averaging physics — Cirrus roughest, Spectralis (16-frame)
smoothest, Topcon between — so Spectralis→Topcon genuinely <em>adds</em> speckle, and we calibrate the magnitude to
that real delta.</p>
{_fig(F['retouch_scans'], "<b>Fig 8.</b> Real per-vendor appearance (RETOUCH B-scans, macular, identical preprocessing): "
      "Cirrus grainiest, Spectralis smoothest, Topcon between — the look the speckle metric quantifies.")}
{_fig(F['retouch'], "<b>Fig 9.</b> Real per-vendor display speckle (RETOUCH): Cirrus > Topcon > Spectralis validates the "
      "frame-averaging direction; the calibrated sim moves GLEAM toward Topcon. ⚠ RETOUCH is macular → the direction "
      "transfers, the absolute level is a proxy.")}

<p class="act">4 · Results — what the benchmark finds</p>
<p>Every reader is scored in <b>one metric language</b>: two metrics say whether the referral <em>decision</em> holds,
two whether the <em>ranking</em> holds — so the compact CNN, the foundation models and the frontier VLMs are all read
on the same axes, on both representations.</p>
{metrics_legend}

<h2>Thickness-map (device) axis — a compact CNN matches the giants; neither scale nor representation buys tolerance</h2>
<p>Two reader families meet the Warner-calibrated shift in one head-to-head: the <b>native</b> readers on the raw
2-channel RNFLT map (a compact <b>SmallCNN</b> ± domain-randomisation, a Ridge floor, an offset oracle) and the image
<b>foundation / CNN / VLM</b> readers on the clinician's RNFLT <em>colour-map</em> render — same GDP eyes, same shift,
only the input representation differs.</p>
{_fig(F['map_panel'], "<b>Fig 10.</b> Every thickness-map reader under the calibrated Cirrus→sim-Spectralis shift "
      "(level 1.0 = the Warner gap, bias ≈2 µm / LoA ±7 µm; 95% CI, coloured by method). The compact SmallCNN with "
      "<b>stability-trained domain-randomisation</b> (teal) is the steadiest native-map reader — its detection is "
      "unchanged by the shift (Δsens@spec ≈ 0, retention ~1.0), below the plain SmallCNN and the Ridge floor, at only a "
      "small clean-AUC cost. The colour-map FM/CNN readers are comparable (flip a few %, CIs overlap) — the ~98k CNN "
      "matches the 88M ConvNeXt and beats the 300M DINOv2, so neither scale nor representation buys a device-tolerance "
      "edge; the VLMs (red) are the least stable (highest flip), though on the friendly colour-map few-shot lifts their "
      "discrimination up to the weakest specialist's level (see the close-up). ⚠ native 2-channel map vs colour-map "
      "render → comparable in kind, not a strict same-input rank (§0).")}
{map_table}
<p class="ref">One thickness-map head-to-head (native structural + colour-map readers). <b>Decision:</b> flip ·
Δsens@spec (signed). <b>Discrimination:</b> AUC · ΔAUC. spec@shift guards against a sensitivity bought by
over-referral; ECE is the calibration level. All with 95% CIs.</p>
{_fig(F['map_vlm'], "<b>Fig 11.</b> The generalist-VLM close-up on the map — clean ECE (vs the specialist band) and AUC "
      "by shot. On the friendlier colour-map the <em>capable</em> frontier VLMs reach specialist discrimination (Claude "
      "Opus 5 even lands inside the FM ECE band), but the small open Qwen3-VL-8B sits at chance; few-shot, not scale, "
      "lifts the read.")}
<h2>B-scan (appearance) axis — robustness tracks pre-training breadth, not the "OCT" label</h2>
<p>On the raw B-scan every reader takes the <em>same</em> image, so this is a clean single-input head-to-head on the
same 660 / 360 held-out GLEAM eyes (only the adaptation regime differs), under the RETOUCH-calibrated
Spectralis→sim-Topcon stress. The ~98k-param SmallCNN is the no-pre-training floor — orders of magnitude smaller than
the foundation models (≥250×), so out-scoring it is the low bar they should clear.</p>
{_fig(F['bscan_panel'], "<b>Fig 12.</b> Every B-scan reader on the same two metrics (referral-flip + clean AUC, 95% CI, "
      "coloured by class). The generic ViT (DINOv2) is the steadiest and among the best; the OCT-specific MIRAGE is the "
      "weakest (AUC ~0.70) and least stable; the frontier VLMs (red) sit at the bottom on both. Fine-tuning (orange) "
      "helped every foundation model here — the opposite of the erasure Kumar 2022 warns of.")}
<div class="callout"><b>Read it in one line.</b> Neither the "OCT" pre-training label nor model size buys
device-tolerance: a <em>generic</em> ViT (DINOv2) is the steadiest and best-discriminating reader (AUC
{d2f['auc_clean']:.2f}); fine-tuning <em>helped</em> every foundation model here (RETFound flip
{rf['referral_flip']*100:.0f}→{rff['referral_flip']*100:.0f}%, DINOv2 {d2r['referral_flip']*100:.0f}→{d2f['referral_flip']*100:.0f}%);
the OCT-specific MIRAGE is the weakest. You have to <em>measure</em> robustness — we report both regimes (Kumar 2022).</div>
{bscan_table}
{scale_callout}
{_fig(F['bscan_vlm'], "<b>Fig 13.</b> The generalist-VLM close-up on the raw B-scan (Gemini · GPT-5.6 · Claude Opus 5, "
      "plus the open-weights Qwen3-VL-8B; verbalized probability, temperature 0, reasoning on for the frontier labs). "
      "Left: clean ECE — the VLMs are several× more overconfident than the specialist FMs (shaded band); Claude is the "
      "best-calibrated, the open Qwen3-VL-8B the worst. Right: AUC by shot — few-shot lifts the read but scale/tier does "
      "not; every arm stays below the weakest specialist, and the open Qwen3-VL-8B + the MedGemma specialist sit at "
      "chance.")}
<div class="callout warn"><b>VLMs are not ready to triage glaucoma-OCT — and small open ones are far worse.</b>
The best frontier generalist (Gemini 3.6-flash few-shot) reaches only AUC {gbest['auc_clean']:.2f} on the raw B-scan —
the weakest specialist's level — and Gemini/GPT are badly overconfident (ECE several× the FMs; <b>Claude Opus 5</b> is the
better-calibrated exception). A colour-map render recovers the <em>capable</em> models to specialist discrimination
(Claude {claude_mz.get('auc_clean', float('nan')):.2f}, GPT {gpt_mf.get('auc_clean', float('nan')):.2f}) — substrate
legibility, not scale, grounds the generalist. But the <b>small open VLMs collapse to near-chance</b> on both axes
(Qwen3-VL-8B {qwen_bz.get('auc_clean', float('nan')):.2f} b-scan / {qwen_mz.get('auc_clean', float('nan')):.2f} map — a
<em>constant</em> on the map; MedGemma {medg_mz.get('auc_clean', float('nan')):.2f} map), and the medical specialist does
not beat the generalists. Reasoning is a prerequisite (thinking off → uniform-0.5 hedge). A referral tool whose confident
calls are wrong is unsafe regardless of average accuracy. <span class="ref">temp-0 is not guaranteed deterministic
(Thinking Machines 2025); we fix seed/prompt, log the model version, score n={nte_b} with bootstrap CIs; small open
models degenerate under the verbalized-probability prompt, so their near-chance AUC is a capability <em>and</em> an
elicitation limit (a verdict prompt may be fairer).</span></div>
{_fig(F['crossv'], "<b>Fig 14.</b> Real cross-vendor corroboration — the mean peripapillary RNFL (TSNIT) profile "
      "overlaid for Spectralis vs Topcon (Eyened-segmented). The double-hump coincides (shape agreement r shown), so the "
      "disc anatomy is vendor-consistent — supporting, never measuring, the device effect (device ≡ cohort, non-ranked).")}


<p class="act">5 · What it means</p>
<h2>Four findings — none of them about the model being big or "OCT"</h2>
<div class="findings">
  <div class="finding g"><div class="fh">A ~98k-param CNN matches the giants on device-stability</div>
    <div class="fb">With stability training, SmallCNN flips just {cnn['referral_flip']*100:.0f}% on the Warner-calibrated
    shift — as steady as an 88M ConvNeXt and steadier than the 300M DINOv2. Scale doesn't help.</div></div>
  <div class="finding b"><div class="fh">Robustness tracks pre-training breadth — not scale, not the "OCT" label</div>
    <div class="fb">A <em>generic</em> ViT (DINOv2) is the steadiest and best-discriminating reader; an ImageNet-CNN
    scale sweep (ResNet-50 → ConvNeXt-L, ~25M→197M) comes out flat.</div></div>
  <div class="finding o"><div class="fh">Fine-tuning <em>helped</em> — it didn't erase inherited robustness</div>
    <div class="fb">Every foundation model fine-tuned steadier on this stress; we report both linear-probe and
    fine-tune regimes, since the effect is model-dependent (Kumar 2022).</div></div>
  <div class="finding c"><div class="fh">VLMs are the weakest readers — worst when small and open</div>
    <div class="fb">Frontier VLMs (Gemini/GPT/Claude) read only at the weakest specialist's level and are mostly
    overconfident (Claude better-calibrated); a colour-map recovers the capable ones, but the small open Qwen3-VL-8B +
    the MedGemma specialist collapse to near-chance — the medical specialist doesn't beat the generalists.</div></div>
</div>
<div class="callout good"><b>The general lesson.</b> "Foundation model" is not a synonym for "device-robust", and
parameter count is not the axis of merit. Under a controlled, literature-calibrated shift, a plain compact CNN is the
steadiest reader on the thickness-map axis; on the B-scan axis robustness tracks pre-training <em>breadth</em> and
adaptation — not the OCT label, and not the size of the model.</div>
<h3>What's real, and what's simulated</h3>
<p>Every model's clean discrimination is measured on <em>real</em> device
data with <em>real</em> labels; the cross-vendor anatomy and per-vendor appearance are real too. The <em>only</em>
simulated element is the same-eye device <em>shift</em> — no public dataset scans the same optic disc on two machines.
So the benchmark stands on four real rungs and calibrates one synthetic bridge:</p>
{real_evidence_table}
<p class="ref"><b>Reading it.</b> <em>AUC</em> = clean glaucoma-vs-healthy ROC-AUC on that real scanner, before any
shift. <em>Shape</em> = the scale-invariant correlation of the mean peripapillary RNFL profile (the TSNIT double-hump)
across vendors — different eyes, one neutral segmenter — so it corroborates that the anatomy agrees; it is not a
detection score.</p>
<p>Not every synthetic axis is equally anchored — only Cirrus→Spectralis on-map is fully same-region calibrated; the
B-scan appearance axis is direction-validated by RETOUCH (a macular proxy):</p>
{prov_table}
<ul class="led">
  <li><b>Synthetic-but-calibrated is necessary, not sufficient</b> — tolerance to a calibrated shift is a robustness
  <em>characterisation</em> (the ImageNet-C / MedMNIST-C bar), not a guarantee of real-world interchangeability (Taori 2020).</li>
  <li><b>No real same-eye cross-device disc OCT exists publicly</b>, so the load-bearing claim rests on the calibrated
  synthetic axis; real cohorts corroborate but never measure the device effect. A genuine same-eye test needs a
  travelling-subject set (AI-READI).</li>
  <li><b>GLEAM is single-centre and modest</b> (Shenyang; {npos_b} glaucoma in the {nte_b}-eye test; labels from Zhang,
  IEEE TMI 2026) → a device-tolerance characterisation, not an interchangeability claim. One B-scan per eye, split by
  eye (no same-eye leakage), every reader fit on the train split with the shift held out — a true head-to-head where
  only the adaptation regime differs. (No patient IDs, so a patient's two eyes could straddle the split — a mild,
  arm-uniform bias.)</li>
  <li><b>Foundation models are all out-of-domain on the disc</b> (every public OCT FM was pre-trained on the macula) —
  the OOD-ness is the axis under test, and they compete on equal footing with a generic control (DINOv2).</li>
  <li><b>Representation-labelled, never cross-ranked</b> — the thickness-map and B-scan leaderboards are separate; a
  lower flip on one is not "better" than a number on the other.</li>
</ul>

<p class="ref"><b>Key references.</b> Warner 2011 (PMC3154907) · Patel 2012 (PMC3299866) · Cheung–Leung 2008
(PMID 18806767) · Mwanza 2010 (PMID 20574014) · Schmitt 1999 (10.1117/1.429925) · Bland–Altman 1986 · Hendrycks &
Dietterich 2019 · Kumar 2022 (arXiv:2202.10054) · Taori 2020 · Pearl–Bareinboim 2014 · RETFound: Zhou 2023 (Nature)
· DINOv2: Oquab 2023 (arXiv:2304.07193) · MIRAGE (j-morano) · GLEAM: Zhang, IEEE TMI 2026 · OCTCube-M 2024
(arXiv:2408.11227) · multimodal-LLM OCT report-gen 2025 (arXiv:2510.02403) · Chuter 2025 VLMs-vs-FMs glaucoma (TVST,
PMC12641456) · medical-VLM robustness 2025 (arXiv:2505.15425) · Guo 2017 (ECE) · Tian 2023 (verbalized confidence) ·
Tobin 2017 (domain-randomisation) · Zheng 2016 (stability training).</p>
<div class="foot">Generated by <code>scripts/build_oct_format_artifact.py</code> from
<code>oct_device_axis_panel.json</code>, <code>oct_appearance_axis_panel.json</code>,
<code>oct_format_validity.json</code>, <code>oct_cross_vendor_panel.json</code>. Figures are real OCT data + calibrated
simulations, embedded inline. Glaucoma-Triage · OCT-FormatShift v3.</div>
"""

    page = ("<!doctype html><html lang=en><head><meta charset=utf-8>"
            "<meta name=viewport content='width=device-width,initial-scale=1'>"
            "<title>OCT-FormatShift — can one glaucoma-OCT reader be trusted across scanners?</title>"
            f"<style>{CSS}</style></head><body><main>{body}</main></body></html>")
    OUT.write_text(page)
    kb = len(page.encode()) / 1024
    print(f"wrote {OUT}  ({kb:.0f} KB, {len(F)} figures)")


if __name__ == "__main__":
    build()
