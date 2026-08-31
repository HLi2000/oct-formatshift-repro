#!/usr/bin/env python
"""Submission-grade figures for the OCT-FormatShift / VLM4RWD paper (papers/oct-formatshift-vlm4rwd/figures/).

A brave editorial redesign of the report figures — NOT a restyle. Reads the committed panels (no re-scoring, no
model calls) and renders a small, curated, colorblind-safe set with NO in-figure titles (the LaTeX caption carries
them), grayscale-legible (colour + marker/hatch), column-width, vector PDF:

  Fig 1  overview     — the two device representations of one eye, clean vs the calibrated same-eye shift, for a
                        clear healthy and a clear glaucomatous eye, beside the real external rung (the setup figure).
  Fig 2  landscape    — every reader on AUC x (decision-flip, ECE), both axes, coloured by class: FMs trustworthy,
                        VLMs weak+overconfident on the raw B-scan, small open/specialist at chance.
  Fig 3  mechanism    — F3, 2x3 (row per axis): decision-flip tracks the score-distribution slide (not ΔAUC) on the
                        B-scan axis and pushes negatives across the fixed threshold (over-referral); the map axis
                        stays clustered near zero (stable).
  appendix figA3      — reliability diagram (VLM vs FM confidence-accuracy).
  appendix figA5      — subgroup-robustness forest (decision-flip by severity/demographics, 95% CIs; C4).
  appendix A1/A2      — map-shift calibration (Bland-Altman/sector-LoA/dose); real per-vendor RETOUCH speckle + TSNIT.
  appendix figA4      — the grounding slope plot (B-scan -> thickness-map jump; demoted from the main text).

    .venv/bin/python scripts/build_paper_submission_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import build_oct_format_artifact as A  # noqa: E402  (classification/label helpers + loaders/transforms)
from paper_naming import disp  # noqa: E402  (canonical reader display names, shared with the tables)
from glaucoma_triage.config import data_dir  # noqa: E402
A.ROOT = data_dir().parent                                          # data lives in the main repo tree

FIGDIR = ROOT / "papers" / "oct-formatshift-vlm4rwd" / "figures"
REPORTS = ROOT / "reports"

# ── Okabe–Ito colorblind-safe palette (+ neutral grey) ────────────────────────────────────────────────
OI = dict(black="#000000", orange="#E69F00", skyblue="#56B4E9", green="#009E73",
          yellow="#F0E442", blue="#0072B2", vermillion="#D55E00", purple="#CC79A7", grey="#8a8a8a")

# reader class -> (colour, marker, label). Grayscale-legible via distinct markers as well as hue.
CLASS = {
    "fm":       (OI["blue"],      "o", "OCT/generic FM"),
    "cnn":      (OI["skyblue"],   "s", "ImageNet CNN"),
    "scratch":  (OI["green"],     "D", "98k CNN (from scratch)"),
    "floor":    (OI["grey"],      "x", "linear floor / oracle"),
    "vlm_closed": (OI["vermillion"], "^", "closed frontier VLM"),
    "vlm_open":   (OI["orange"],     "v", "open-weights VLM"),
    "vlm_spec":   (OI["purple"],     "P", "medical-specialist VLM"),
}

plt.rcParams.update({
    "figure.facecolor": "white", "savefig.facecolor": "white", "savefig.dpi": 300,
    "font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 7.2,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.linewidth": 0.7,
    "axes.edgecolor": "#444", "xtick.color": "#333", "ytick.color": "#333",
    "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6, "pdf.fonttype": 42,
    "mathtext.fontset": "stix",  # Times-matched serif math (body font is Times New Roman) -> consistent typography
})


def _classify(m: str) -> str:
    if m.startswith(("gemini", "openai", "claude")):
        return "vlm_closed"
    if m.startswith("qwen3vl"):
        return "vlm_open"
    if m.startswith("medgemma"):
        return "vlm_spec"
    if m.startswith(("retfound", "dinov2", "mirage")):
        return "fm"
    if m.startswith(("resnet", "convnext")):
        return "cnn"
    if m in ("supervised", "fullres_cnn", "fullres_cnn_dr") or "SmallCNN" in m:
        return "scratch"
    return "floor"


def _save(fig, stem):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"{stem}.{ext}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  · {stem}")


def _load():
    app = json.load(open(REPORTS / "oct_appearance_axis_panel.json"))["panel"]
    mp = json.load(open(REPORTS / "oct_map_fm_panel.json"))["panel"]
    dev = json.load(open(REPORTS / "oct_device_axis_panel.json"))
    valid = json.load(open(REPORTS / "oct_format_validity.json"))
    cv = json.load(open(REPORTS / "oct_cross_vendor_panel.json"))
    return app, mp, dev, valid, cv


# ── Fig 1 — overview: two representations × clean/shift × healthy/glaucoma + real rung ─────────────────
def fig1_overview():
    from glaucoma_triage.oct_format import bscan_transforms as BT, canonical, loaders, transforms
    maps, y = loaders.load_harvard_gdp_maps(limit=1000)
    mh, mg = 671, 400                                               # committed by-eye exemplars
    X, yb = loaders.load_gleam(balanced=True, limit=400)
    gh, gg = 108, 147
    shm = lambda m, s: transforms.scanner_offset(m, np.random.default_rng(s), 1.0, pair=transforms.DEFAULT_PAIR)
    shb = lambda im, s: BT.apply_bscan_device_shift(np.asarray(im, np.float32), np.random.default_rng(s), 1.0,
                                                    ref="spectralis", target="topcon")
    fig = plt.figure(figsize=(7.9, 2.95))
    # (a) BENCHMARK: per eye clean|sim|Delta (maps, 6 cols) over clean|sim (B-scans, 4 cols) | spacer+cbars | (b) REAL
    outer = fig.add_gridspec(1, 3, width_ratios=[6.0, 1.15, 1.55], wspace=0.0,
                             left=0.055, right=0.995, top=0.84, bottom=0.07)
    bench = outer[0, 0].subgridspec(2, 1, hspace=0.36)
    maprow = bench[0].subgridspec(1, 6, wspace=0.08)
    brow = bench[1].subgridspec(1, 4, wspace=0.08)
    vmax = float(np.nanpercentile(np.concatenate([maps[mh].ravel(), maps[mg].ravel()]), 99))
    cmap = plt.get_cmap("viridis").copy(); cmap.set_bad("#eeede8")
    dcmap = plt.get_cmap("coolwarm").copy(); dcmap.set_bad("#eeede8")
    DVMAX = 10.0  # +/- um diverging scale for the sim-clean difference (spans the +/-7.68 um calibrated LoA)
    vlast = [None]; dlast = [None]
    def _bare(ax, t):
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(t, fontsize=7.0, labelpad=1)
        for s in ("top", "right", "left", "bottom"):
            ax.spines[s].set_visible(False)
    # trim the outer all-NaN frame: under bilinear interpolation the masked border (fill value 0) otherwise bleeds a
    # stray viridis-minimum sliver into the panel edge at mid-height. 2px of the extreme periphery is outside the scan.
    _trim = lambda a: np.asarray(a, float)[2:-2, 2:-2]
    def mpc(ax, m, t):
        vlast[0] = ax.imshow(np.ma.masked_invalid(_trim(m)), cmap=cmap, vmin=0, vmax=vmax, interpolation="bilinear"); _bare(ax, t)
    def mpd(ax, d, t):
        dlast[0] = ax.imshow(np.ma.masked_invalid(_trim(d)), cmap=dcmap, vmin=-DVMAX, vmax=DVMAX, interpolation="bilinear"); _bare(ax, t)
    def bp_(ax, im, t): A._imshow_bscan(ax, np.asarray(im, float), ""); ax.set_xlabel(t, fontsize=7.0, labelpad=1)
    ch = maps[mh]; sh = shm(ch, 31); cg = maps[mg]; sg = shm(cg, 4)   # independent per-eye seeds; each a representative draw (mean~bias, sup/inf-wider)
    mc = [fig.add_subplot(maprow[0, i]) for i in range(6)]
    mpc(mc[0], ch, "clean (Cirrus)"); mpc(mc[1], sh, "sim Spectralis"); mpd(mc[2], sh - ch, r"$\Delta$ (sim$-$clean)")
    mpc(mc[3], cg, "clean (Cirrus)"); mpc(mc[4], sg, "sim Spectralis"); mpd(mc[5], sg - cg, r"$\Delta$ (sim$-$clean)")
    bc = [fig.add_subplot(brow[0, i]) for i in range(4)]
    bp_(bc[0], X[gh], "clean (Spectralis)"); bp_(bc[1], shb(X[gh], 3), "sim Topcon")
    bp_(bc[2], X[gg], "clean (Spectralis)"); bp_(bc[3], shb(X[gg], 4), "sim Topcon")
    # two stacked colorbars in the spacer: viridis (RNFL) for clean/sim, coolwarm (shared) for the Delta panels
    Pm = [ax.get_position() for ax in mc]; Pb = [ax.get_position() for ax in bc]
    # (b) real column: place the TSNIT against the map row and the two real B-scans against the B-scan row, so the
    # real Spectralis/Topcon pair aligns vertically with (a)'s B-scan panels (taller than a nested gridspec allows).
    rc = outer[0, 2].get_position(fig); rx0 = rc.x0 + 0.012; rw = rc.width - 0.020
    a5 = fig.add_axes([rx0, Pm[0].y0, rw, Pm[0].y1 - Pm[0].y0]); _real_tsnit(a5)
    bgap = 0.28 * (Pb[0].y1 - Pb[0].y0); bhh = (Pb[0].y1 - Pb[0].y0 - bgap) / 2   # gap leaves room for the Spectralis label
    _real_bscan_pair(fig.add_axes([rx0, Pb[0].y1 - bhh, rw, bhh]), fig.add_axes([rx0, Pb[0].y0, rw, bhh]))
    top = Pm[0].y1; bot = Pm[0].y0; h = top - bot; x0 = Pm[5].x1 + 0.012
    cax1 = fig.add_axes([x0, bot + 0.56 * h, 0.008, 0.44 * h])
    cb1 = fig.colorbar(vlast[0], cax=cax1); cb1.ax.tick_params(labelsize=5.4, length=1.4); cb1.outline.set_linewidth(0.4)
    fig.text(x0 + 0.004, top + 0.015, "RNFL\n(µm)", fontsize=5.4, ha="center", va="bottom", color="#555")
    cax2 = fig.add_axes([x0, bot, 0.008, 0.40 * h])
    cb2 = fig.colorbar(dlast[0], cax=cax2); cb2.ax.tick_params(labelsize=5.4, length=1.4); cb2.outline.set_linewidth(0.4)
    cb2.set_ticks([-DVMAX, 0, DVMAX])
    fig.text(x0 + 0.004, bot + 0.40 * h + 0.004, r"$\Delta$ (µm)", fontsize=5.4, ha="center", va="bottom", color="#555")
    # titles / axis labels
    p5 = a5.get_position(); yt = top
    acx = (Pm[0].x0 + Pm[5].x1) / 2; bcx = (p5.x0 + p5.x1) / 2; divx = p5.x0 - 0.02
    fig.add_artist(Line2D([divx, divx], [0.06, yt + 0.10], color="#cccccc", lw=0.8, transform=fig.transFigure))
    fig.text(acx, yt + 0.125, "(a) Benchmark: calibrated same-eye shift", ha="center", va="bottom",
             fontsize=8.2, weight="bold")
    fig.text(bcx, yt + 0.125, "(b) Real validation", ha="center", va="bottom", fontsize=8.2, weight="bold")
    fig.text((Pm[0].x0 + Pm[2].x1) / 2, yt + 0.025, "healthy eye", ha="center", va="bottom", fontsize=7.0,
             style="italic", color="#555")
    fig.text((Pm[3].x0 + Pm[5].x1) / 2, yt + 0.025, "glaucoma eye", ha="center", va="bottom", fontsize=7.0,
             style="italic", color="#555")
    fig.text(0.010, (Pm[0].y0 + Pm[0].y1) / 2, "thickness\nmap axis", va="center", ha="left", fontsize=7.4, rotation=90)
    fig.text(0.010, (Pb[0].y0 + Pb[0].y1) / 2, "B-scan\naxis", va="center", ha="left", fontsize=7.4, rotation=90)
    _save(fig, "fig1_overview")


def _real_tsnit(ax):
    p = A.ROOT / "data/derived/eyened_cprnfl.npz"
    if p.exists():
        d = np.load(p); x = np.linspace(0, 360, d["spectralis"].shape[1])
        for v, c in (("spectralis", OI["orange"]), ("topcon", OI["green"])):
            P = np.asarray(d[v], float); Z = (P - P.mean(1, keepdims=True)) / (P.std(1, keepdims=True) + 1e-9)
            ax.plot(x, Z.mean(0), color=c, lw=1.1)
        ax.set_xlim(0, 360); ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("real cross-vendor", fontsize=7.2, labelpad=1)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)


def _norm_bscan(im, gamma=1.0, blur=0.0):
    """Percentile-window a B-scan to a common brightness, with optional mild gamma + blur (display only)."""
    from PIL import Image, ImageFilter
    pil = im.convert("L") if hasattr(im, "convert") else Image.fromarray(np.asarray(im, np.uint8)).convert("L")
    if blur:
        pil = pil.filter(ImageFilter.GaussianBlur(blur))
    a = np.asarray(pil, np.float32)
    lo, hi = np.percentile(a, 2), np.percentile(a, 98)
    a = np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)
    return a ** gamma


def _cands(ret, vendor, k=8):
    for v in (vendor, vendor.lower(), vendor.upper()):
        imgs = [im for im in (ret.get(v) or []) if getattr(im, "size", 0) > 100]
        if imgs:
            return imgs[:k]
    return []


def _band_profile(im):
    """Row-mean intensity profile (retinal-band position/curvature signature), normalised to length 128."""
    rows = _norm_bscan(im).mean(1)
    p = np.interp(np.linspace(0, 1, 128), np.linspace(0, 1, len(rows)), rows)
    return (p - p.mean()) / (p.std() + 1e-6)


def _crop_aspect(a, target=1.65):
    """Crop to a fixed width:height (no imshow rescale-stretch), *centred on the retinal band* (its brightness
    centroid), not the geometric centre — so the informative layers, not empty vitreous/choroid, fill the frame.
    A moderate target leaves a bit of vitreous/choroid margin above and below the band."""
    h, w = a.shape[:2]
    nh = int(round(w / target))
    if nh >= h:                                                    # already wide enough -> crop width, band-centred
        nw = int(round(h * target)); col = a.mean(0)
        cx = int(np.clip(np.sum(np.arange(w) * col) / (col.sum() + 1e-9) - nw // 2, 0, max(0, w - nw)))
        return a[:, cx:cx + nw]
    row = a.mean(1)                                                # vertical brightness profile -> band centroid
    cy = int(np.clip(np.sum(np.arange(h) * row) / (row.sum() + 1e-9) - nh // 2, 0, h - nh))
    return a[cy:cy + nh, :]


def _prep_real(im, dim=1.0, fade=None):
    """Central-crop + normalise; optional brightness dim and a lower-half fade (the noisy choroid fades out)."""
    a = _crop_aspect(_norm_bscan(im)) * dim
    if fade is not None:
        r = np.linspace(0, 1, a.shape[0])
        g = np.where(r > fade, np.interp(r, [fade, 1.0], [1.0, 0.12]), 1.0)
        a = a * g[:, None]
    return np.clip(a, 0, 1)


def _real_bscan_pair(ax_s, ax_t):
    """Real Spectralis vs real Topcon macular B-scans chosen to depict a *similar retinal region* (best-matched
    band profiles), so the real pair mirrors the sim pair. Light percentile normalisation only (no heavy denoise)."""
    ret = A.loaders.load_retouch_sample(per_vendor=8)
    S, T = _cands(ret, "Spectralis"), _cands(ret, "Topcon")
    s, t = (S[0] if S else None), (T[0] if T else None)
    if S and T:                                                    # most anatomically similar cross-vendor pair
        ps = [_band_profile(im) for im in S]; pt = [_band_profile(im) for im in T]
        i, j, _ = max(((i, j, float(np.corrcoef(ps[i], pt[j])[0, 1]))
                       for i in range(len(S)) for j in range(len(T))), key=lambda z: z[2])
        s, t = S[i], T[j]
    ax_s.imshow(_prep_real(s) if s is not None else np.zeros((10, 10)), cmap="gray", aspect="auto", vmin=0, vmax=1)
    ax_t.imshow(_prep_real(t, dim=0.82, fade=0.5) if t is not None else np.zeros((10, 10)),
                cmap="gray", aspect="auto", vmin=0, vmax=1)
    for ax, lab in ((ax_s, "real Spectralis"), (ax_t, "real Topcon")):
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(lab, fontsize=7.2, labelpad=1)
        for spn in ("top", "right", "left", "bottom"):
            ax.spines[spn].set_visible(False)


# ── helpers for the metric-forward main figures ───────────────────────────────────────────────────────
def _lvl(rec):
    """per-arm record from a panel value (per_level 1.0) or a device headline_board entry."""
    return rec["per_level"]["1.0"] if isinstance(rec, dict) and "per_level" in rec else rec


def _present_legend(classes):
    return [Line2D([0], [0], marker=CLASS[c][1], color="w", mfc=CLASS[c][0], mec="white", ms=6, label=CLASS[c][2])
            for c in CLASS if c in classes]


# ── Fig 2 (HERO) — reliability landscape: decision-stability (flip) + calibration (ECE) vs AUC, both axes ─
def fig2_stability(app, mp, dev):
    """Reliability landscape. Every reader at (clean AUC, y) for two reliability metrics x two axes:
    row 1 = decision-stability (decision-flip, lower=steadier), row 2 = calibration (clean ECE, lower=better);
    columns = B-scan | thickness-map. The 98k from-scratch CNN (green diamond) is a first-class arm on both axes."""
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.0), sharex=True, sharey="row")
    present = set()
    metrics = [("referral_flip", "decision-flip  ($\\downarrow$ steadier)", (-0.02, 0.60), "decision-stability"),
               ("ece_clean", "clean ECE  ($\\downarrow$ calibrated)", (-0.01, 0.60), "calibration")]
    cols = [("B-scan", app), ("thickness-map", mp)]
    tags = "abcd"
    for ri, (key, ylab, ylim, mlab) in enumerate(metrics):
        for ci, (axname, panel) in enumerate(cols):
            ax = axes[ri][ci]
            for m, v in panel.items():
                r = _lvl(v); cls = _classify(m); present.add(cls)
                isv = cls in ("vlm_closed", "vlm_open", "vlm_spec")     # VLMs on the top layer
                ax.scatter(r["auc_clean"], r[key], s=22, c=CLASS[cls][0], marker=CLASS[cls][1],
                           edgecolor="white", linewidth=0.35,
                           zorder=5 if isv else (4 if cls == "scratch" else 3))
            ax.set_xlim(0.44, 0.92); ax.set_ylim(*ylim)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if ci == 0:
                ax.set_ylabel(ylab)
            if ri == 1:
                ax.set_xlabel("clean AUC  ($\\rightarrow$ better)")
            # titles consistently top-left, in the headroom above all points (ylim tops raised for this)
            ax.text(0.03, 0.97, f"({tags[ri * 2 + ci]}) {axname} $\\cdot$ {mlab}", transform=ax.transAxes,
                    fontsize=6.7, weight="bold", va="top", ha="left")
    # only annotation kept: MIRAGE, the unstable OCT FM (not identifiable from the legend, which has several FMs)
    rm = _lvl(app["mirage"])
    axes[0][0].annotate("MIRAGE\n(unstable)", (rm["auc_clean"], rm["referral_flip"]),
                        xytext=(rm["auc_clean"] + 0.02, rm["referral_flip"]), fontsize=6.0, color=OI["blue"],
                        ha="left", va="center")
    # (the 98k from-scratch map CNN is now a first-class mp arm ("supervised"), plotted by the main loop above)
    handles = _present_legend(present)
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=6.6,
               bbox_to_anchor=(0.5, -0.01), handletextpad=0.3, columnspacing=1.6)
    fig.subplots_adjust(bottom=0.13, hspace=0.14, wspace=0.06)
    _save(fig, "fig2_stability")


# ── Fig 3 — the grounding twist: B-scan -> thickness map recovers AUC, ECE, and flip together ──────────
def figA4_grounding(app, mp):
    arms = [("gemini_3p6_flash_fewshot_think", "gemini_map_3p6_flash_fewshot_think", "Gemini 3.6 Flash", "vlm_closed"),
            ("openai_5p6_sol_fewshot_think", "openai_map_5p6_sol_fewshot_think", "GPT-5.6 Sol", "vlm_closed"),
            ("claude_opus_5_fewshot_think", "claude_map_opus_5_fewshot_think", "Claude Opus 5", "vlm_closed"),
            ("qwen3vl_8b_fewshot", "qwen3vl_map_8b_fewshot", "Qwen3-VL-8B", "vlm_open"),
            ("medgemma_4b_fewshot", "medgemma_map_4b_fewshot", "MedGemma-1.5-4B", "vlm_spec")]
    metrics = [("auc_clean", "clean AUC  ($\\uparrow$)", (0.44, 0.9)),
               ("ece_clean", "clean ECE  ($\\downarrow$)", (0.0, 0.55)),
               ("referral_flip", "decision-flip  ($\\downarrow$)", (0.0, 0.5))]
    fig, axes = plt.subplots(1, 3, figsize=(6.7, 2.5))
    lines = {}
    # frontier VLMs share the orange triangle (consistent w/ Fig 2); the three companies differ by line style
    vstyle = {"Gemini 3.6 Flash":   (OI["vermillion"], "^", "-"),
              "GPT-5.6 Sol":       (OI["vermillion"], "^", (0, (5, 2))),
              "Claude Opus 5": (OI["vermillion"], "^", (0, (1, 1.4))),
              "Qwen3-VL-8B":   (OI["orange"],     "v", "-"),
              "MedGemma-1.5-4B": (OI["purple"],   "P", "-")}
    for ax, (key, ylab, ylim) in zip(axes, metrics):
        bnv = [_lvl(app[m])[key] for m in app if not A._is_vlm(m)]     # non-VLM readers, PER AXIS
        mnv = [_lvl(mp[m])[key] for m in mp if not A._is_vlm(m)]
        blo, bhi = np.percentile(bnv, [10, 90]); mlo, mhi = np.percentile(mnv, [10, 90])
        ax.fill_between([0, 1], [blo, mlo], [bhi, mhi], color=OI["blue"], alpha=0.10, lw=0)  # 10-90% band per axis
        for bk, mk, lab, cls in arms:
            c, mrk, ls = vstyle[lab]
            bv, mv = _lvl(app[bk])[key], _lvl(mp[mk])[key]
            ln, = ax.plot([0, 1], [bv, mv], color=c, linestyle=ls, lw=1.3, marker=mrk, ms=4, mfc=c, mec="white", mew=0.4)
            lines[lab] = ln
        ax.set_xticks([0, 1]); ax.set_xticklabels(["B-scan", "thickness\nmap"]); ax.set_xlim(-0.12, 1.12)
        ax.set_ylabel(ylab); ax.set_ylim(*ylim)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].text(0.5, 0.885, "non-VLM readers (10-90%)", fontsize=6.0, color=OI["blue"], ha="center", va="top")
    fig.legend(list(lines.values()), list(lines.keys()), loc="lower center", ncol=5, frameon=False,
               fontsize=6.6, bbox_to_anchor=(0.5, -0.04), handletextpad=0.4, columnspacing=1.0, handlelength=3.4)
    fig.subplots_adjust(bottom=0.28, wspace=0.42)
    _save(fig, "figA4_grounding")


# ── Fig 3 (main) — F3 mechanism: flip is set by the coherent score-distribution SLIDE, not by ΔAUC ─────
_CLS_STYLE = {"OCT FM": (OI["blue"], "o"), "generic FM": (OI["blue"], "o"), "ImageNet CNN": (OI["skyblue"], "s"),
              "from-scratch CNN": (OI["green"], "D"), "frontier VLM": (OI["vermillion"], "^"),
              "open VLM": (OI["orange"], "v"), "specialist VLM": (OI["purple"], "P"), "other": (OI["grey"], "x")}


def fig3_mechanism():
    """2x3 mechanism figure, one row per shift axis (top: B-scan appearance; bottom: thickness map). Columns:
    (1) decision-flip vs distribution-slide magnitude |mean Δp|, (2) decision-flip vs |ΔAUC|, (3) one exemplar's
    true-negative score density, clean vs shifted, across the fixed 0.5 threshold. On the B-scan axis flip tracks the
    slide (rho=+0.68), not the discrimination change (rho=+0.09), and the exemplar's negatives slide past 0.5
    (over-referral). On the thickness-map axis every arm clusters near zero slide/ΔAUC/flip (stable, same axis
    limits), so the exemplar's clean and shifted densities nearly coincide. Built entirely from
    reports/shift_mechanism.json (flip_up/flip_dn cached per arm); no npz, no model call. Model names are dropped
    from the scatters (the trend is the message)."""
    SYS = json.loads((REPORTS / "shift_mechanism.json").read_text())["systematic"]
    XLIM_SLIDE, XLIM_AUC, YLIM = (-0.015, 0.34), (-0.004, 0.085), (-0.03, 0.52)  # matched across rows

    def _scatter(ax, arms, xkey, rho, xlim):
        for a, r in arms.items():
            col, mk = _CLS_STYLE.get(r["cls"], _CLS_STYLE["other"])
            ax.scatter(abs(r[xkey]), r["flip"], color=col, marker=mk, s=22, edgecolor="white", linewidth=0.4,
                       alpha=0.9, zorder=3)
        ax.set_xlim(*xlim); ax.set_ylim(*YLIM)
        ax.text(0.96, 0.94, rf"$\rho={rho:+.2f}$", transform=ax.transAxes, fontsize=7.2, ha="right", va="top")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    # column (c): flip DIRECTION (newly-refer up vs newly-do-not-refer down), one reader per class plus the workshop's
    # frontier VLMs. Shows appearance-keyed readers over-refer one-directionally, the generic FM is robust, and the
    # VLMs churn model-specifically -- while every reader is near-flat on the map. Bars use non-class colours (red /
    # grey) to avoid clashing with the reader-class marker colours in (a,b). MIRAGE (OCT-FM) is not run on maps -> n/a.
    OVER, UNDER = "#CC3311", "#999999"
    READERS = [("dinov2_ft", "dinov2_ft"),
               ("retfound", None),
               ("mirage", None),
               ("resnet50_ft", "resnet50_ft"),
               ("convnext_base", "convnext_base"),
               ("supervised", "supervised"),
               ("gemini_3p6_flash_fewshot_think", "gemini_map_3p6_flash_fewshot_think"),
               ("openai_5p6_sol_fewshot_think", "openai_map_5p6_sol_fewshot_think"),
               ("claude_opus_5_fewshot_think", "claude_map_opus_5_fewshot_think")]
    N_NONVLM = 6                          # DINOv2, RETFound, MIRAGE, ResNet, ConvNeXt, SmallCNN | 3 VLMs

    def _flipdir(ax, axis, show_xlabels, show_legend):
        arms = SYS[axis]["arms"]
        for xi, (bk, mk) in enumerate(READERS):
            key = mk if axis == "map" else bk
            if key is None:                                        # OCT-FM: no map render
                ax.text(xi, 0.03, "n/a", rotation=90, ha="center", va="bottom", fontsize=5.2, color="#aaa")
                continue
            up = arms[key]["flip_up"]                              # newly referred (toward over-referral)
            dn = arms[key]["flip_dn"]                              # newly not-referred (toward under-referral)
            ax.bar(xi, up, width=0.66, color=OVER, edgecolor="white", lw=0.4, zorder=3,
                   label="newly refer" if (show_legend and xi == 0) else None)
            ax.bar(xi, -dn, width=0.66, color=UNDER, edgecolor="white", lw=0.4, zorder=3,
                   label="newly do-not-refer" if (show_legend and xi == 0) else None)
        ax.axhline(0, color="#444", lw=0.7, zorder=2)
        ax.axvline(N_NONVLM - 0.5, color="#cfcfcf", lw=0.6, ls=":", zorder=1)   # non-VLM | VLM divider
        ax.set_ylim(-0.20, 0.52); ax.set_xlim(-0.6, len(READERS) - 0.4)
        ax.set_xticks(range(len(READERS)))
        ax.set_xticklabels([disp(bk, "fig") for bk, _ in READERS] if show_xlabels else [], fontsize=5.4,
                           rotation=30, ha="right")
        ax.set_yticks([-0.15, 0, 0.15, 0.30, 0.45]); ax.set_yticklabels(["0.15", "0", "0.15", "0.30", "0.45"])
        ax.grid(axis="x", visible=False)
        if show_legend:
            ax.legend(fontsize=5.6, frameon=False, loc="upper right", handlelength=1.0, handletextpad=0.3,
                      labelspacing=0.2)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    fig, axes = plt.subplots(2, 3, figsize=(7.6, 4.6), gridspec_kw={"width_ratios": [1, 1, 1.7]})
    rows = [("bscan", "B-scan appearance"), ("map", "thickness map")]
    for i, (axis, rlab) in enumerate(rows):
        arms, summ = SYS[axis]["arms"], SYS[axis]["summary"]
        _scatter(axes[i, 0], arms, "mean_dp", summ["rho_flip_vs_slide"], XLIM_SLIDE)
        _scatter(axes[i, 1], arms, "delta_auc", summ["rho_flip_vs_delta_auc"], XLIM_AUC)
        _flipdir(axes[i, 2], axis, show_xlabels=(i == 1), show_legend=(i == 0))
        axes[i, 0].set_ylabel("decision-flip"); axes[i, 2].set_ylabel("flip fraction")
        axes[i, 0].annotate(rlab, xy=(-0.34, 0.5), xycoords="axes fraction", rotation=90, ha="center", va="center",
                            fontsize=8.5, fontweight="bold")
    for j, xl in enumerate([r"distribution slide  $|\overline{\Delta p}|$",
                            r"discrimination change  $|\Delta\mathrm{AUC}|$", ""]):  # col (c) named by its x-ticks
        axes[1, j].set_xlabel(xl)
    # shared class legend for the scatters (a,b); OCT & generic FM share the blue circle, as in Fig 2
    seen = [("OCT/generic FM", "OCT FM"), ("ImageNet CNN", "ImageNet CNN"), ("from-scratch CNN", "from-scratch CNN"),
            ("frontier VLM", "frontier VLM"), ("open VLM", "open VLM"), ("specialist VLM", "specialist VLM")]
    handles = [Line2D([0], [0], marker=_CLS_STYLE[k][1], color="w", mfc=_CLS_STYLE[k][0], mec="white", ms=6, label=lab)
               for lab, k in seen]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=6.0,
               bbox_to_anchor=(0.5, -0.015), handletextpad=0.2, columnspacing=0.8)
    for j, pl in enumerate(["(a)", "(b)", "(c)"]):          # column panel labels (the caption refers to a/b/c)
        axes[0, j].annotate(pl, xy=(0.0, 1.04), xycoords="axes fraction", ha="left", va="bottom",
                            fontsize=9, fontweight="bold")
    fig.subplots_adjust(left=0.11, right=0.98, top=0.93, bottom=0.15, wspace=0.40, hspace=0.24)
    _save(fig, "fig3_mechanism")


# ── Appendix — reliability diagram (calibration detail): VLM vs FM confidence vs empirical accuracy ────
def _reliability_probs(npz):
    """Per-eye (clean prob, label) for one arm from its committed .npz (no encoder, no GPU). VLM arms store
    ``clean_prob`` directly; FM/CNN arms store cached features + the seed-0 split, so we reproduce the panel's clean
    probabilities with the SAME frozen probe (``fm_contender._probe``) fit on the cached train features."""
    from glaucoma_triage.oct_format import fm_contender as FM
    d = np.load(npz, allow_pickle=True)
    y = np.asarray(d["labels"]).astype(int)
    if "clean_prob" in d:
        return np.asarray(d["clean_prob"], float), y
    X = np.asarray(d["clean"], float); tr = np.asarray(d["tr"]); te = np.asarray(d["te"])
    ytr = y[tr]
    pipe = FM._probe(int(ytr.sum()), int((ytr == 0).sum())).fit(X[tr], ytr)
    return pipe.predict_proba(X[te])[:, 1], y[te]


def figA3_reliability():
    # curves for arms with enough distinct probabilities; the near-chance small models (~3-9 distinct values) are
    # shown as single (mean-confidence, accuracy) points -- a degenerate line would misrepresent them.
    # one representative per class (FM / frontier VLM), spanning the calibration spectrum; the near-chance small
    # models (open, specialist) are single points -- they hedge to ~3-9 distinct probabilities so a line would be fake.
    # group contrast: ALL frontier-VLM arms (few-shot) vs FM + CNN arms, one thin class-coloured curve each; the
    # systematic VLM band below the diagonal vs the FM/CNN band on it is stronger than any one-vs-one comparison.
    # frontier VLMs share the vermillion frontier-VLM class colour (as in Figs 2-3) and are told apart by line
    # style + marker; this avoids reusing the open (orange) / specialist (purple) class colours from the other figures.
    groups = [("Gemini 3.6 Flash", OI["vermillion"], "-", "o", ["gemini_3p6_flash_fewshot_think"]),
              ("GPT-5.6 Sol", OI["vermillion"], "--", "s", ["openai_5p6_sol_fewshot_think"]),
              ("Claude Opus 5", OI["vermillion"], ":", "^", ["claude_opus_5_fewshot_think"]),
              ("OCT/generic FM", OI["blue"], "-", "o", ["dinov2_reg", "retfound", "mirage"]),
              ("ImageNet CNN", OI["skyblue"], "-", "D", ["convnext_base", "resnet50"])]
    fig, ax = plt.subplots(figsize=(3.8, 3.2))
    ax.plot([0.5, 1], [0.5, 1], ls=":", color="#666", lw=1.0, zorder=1)              # perfect calibration
    seen = set()
    for gname, c, ls, mk, arms in groups:
        for a in arms:
            p = A.ROOT / "data/derived" / f"{a}_contender.npz"
            if not p.exists():
                continue
            prob, y = _reliability_probs(p)
            conf = np.maximum(prob, 1 - prob); correct = ((prob >= 0.5).astype(int) == y).astype(int)
            order = np.argsort(conf); xs, ys = [], []
            for chunk in np.array_split(order, 5):                        # equal-count bins -> stable per-bin accuracy
                if len(chunk) >= 10:
                    xs.append(conf[chunk].mean()); ys.append(correct[chunk].mean())
            lab = gname if gname not in seen else None; seen.add(gname)
            ax.plot(xs, ys, color=c, ls=ls, marker=mk, lw=1.1, alpha=0.75, ms=2.6, mfc=c, mec="none",
                    label=lab, zorder=3)
    ax.text(0.55, 0.60, "perfect calibration", fontsize=5.6, color="#777", rotation=33)
    ax.annotate("frontier VLMs overconfident\n(accuracy $<$ confidence)", (0.82, 0.42), fontsize=6.0,
                color=OI["vermillion"], ha="center")
    ax.set_xlabel("mean predicted confidence"); ax.set_ylabel("empirical accuracy")
    ax.set_xlim(0.49, 1.01); ax.set_ylim(0.30, 1.02)
    ax.legend(fontsize=6.2, loc="upper left", frameon=False, handletextpad=0.3, labelspacing=0.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    _save(fig, "figA3_reliability")


# ── Appendix A1 — map-shift calibration validity, rebuilt NATIVELY in the submission style ────────────
def _a1_bland(valid):
    X, _ = A.loaders.load_harvard_gf_maps(limit=400)
    st = A.CO.map_difference_stats(X, "cirrus_to_spectralis", np.random.default_rng(7), severity=1.0)["global"]
    diffs = np.asarray(st["diffs"], float); means = np.linspace(60, 105, len(diffs))
    bias, loa = st["bias"], 1.96 * st["sd"]
    pub = valid["V1a_realism"]["cirrus_to_spectralis"]; pb, pl = pub["target_bias"], pub["published_loa_half"]
    fig, ax = plt.subplots(figsize=(2.65, 2.4))
    ax.axhspan(pb - pl, pb + pl, color=OI["skyblue"], alpha=0.22, lw=0)
    ax.scatter(means, diffs, s=7, color=OI["blue"], alpha=0.40, edgecolor="none", zorder=3)
    ax.axhline(bias, color=OI["vermillion"], lw=1.2, zorder=4)
    for yv in (bias - loa, bias + loa):
        ax.axhline(yv, color=OI["vermillion"], ls="--", lw=1.0, zorder=4)
    ax.axhline(0, color="#999", lw=0.7, ls=":")
    ax.text(0.97, 0.97, f"Warner LoA {pb:+.1f}$\\pm${pl:.1f}", transform=ax.transAxes, ha="right", va="top",
            fontsize=5.8, color=OI["blue"])
    ax.text(0.97, 0.04, f"sim {bias:+.2f}$\\pm${loa:.1f}", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=5.8, color=OI["vermillion"])
    ax.set_xlabel("mean RNFL thickness (µm)"); ax.set_ylabel("Spectralis $-$ Cirrus (µm)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    _save(fig, "figA1a_bland_altman")


def _a1_sector(valid):
    v = valid["V1a_structural"]; s = v["sector_loa_half_um"]; pub = v["published_loa_half_um"]
    order = [("Sup", s["s"]), ("Inf", s["i"]), ("Nas", s["n"]), ("Tmp", s["t"])]
    fig, ax = plt.subplots(figsize=(2.65, 2.4))
    ax.bar([o[0] for o in order], [o[1] for o in order],
           color=[OI["vermillion"], OI["vermillion"], OI["grey"], OI["grey"]], width=0.62)
    for i, (_, val) in enumerate(order):
        ax.text(i, val + 0.12, f"{val:.1f}", ha="center", fontsize=6)
    ax.axhline(pub, color=OI["blue"], ls="--", lw=1.0)
    ax.text(3.45, pub + 0.15, f"Warner global {pub:.1f}", fontsize=5.6, color=OI["blue"], ha="right", va="bottom")
    ax.set_ylabel("LoA half-width (µm)"); ax.set_ylim(0, max(o[1] for o in order) + 1.7)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    _save(fig, "figA1b_sector_loa")


def _a1_dose(valid):
    rows = sorted(valid["V3_monotonicity"]["rows"], key=lambda r: r["level"])
    lv = [r["level"] for r in rows]; flip = [r["flip_rate"] for r in rows]; drift = [r["read_drift"] for r in rows]
    fig, ax = plt.subplots(figsize=(2.65, 2.4))
    ax.plot(lv, flip, "-o", color=OI["vermillion"], ms=3, lw=1.2)
    ax.set_xlabel("shift dose (severity)"); ax.set_ylabel("decision-flip rate", color=OI["vermillion"])
    ax.tick_params(axis="y", labelcolor=OI["vermillion"]); ax.set_ylim(bottom=0)
    ax2 = ax.twinx(); ax2.plot(lv, drift, "--s", color=OI["blue"], ms=3, lw=1.2)
    ax2.set_ylabel("read-drift (µm)", color=OI["blue"]); ax2.tick_params(axis="y", labelcolor=OI["blue"])
    ax2.set_ylim(bottom=0)
    ax.plot([], [], "-o", color=OI["vermillion"], ms=3, label="decision-flip")
    ax.plot([], [], "--s", color=OI["blue"], ms=3, label="read-drift")
    ax.legend(loc="upper left", fontsize=5.8, frameon=False)
    ax.spines["top"].set_visible(False); ax2.spines["top"].set_visible(False)
    _save(fig, "figA1c_dose_response")


# ── Appendix A2 — appearance/anatomy validity: reuse the vetted renderers, cleaned to submission style ─
def appendix(valid, cv):
    _a1_bland(valid); _a1_sector(valid); _a1_dose(valid)

    def cap(render):
        buf = {}
        orig = A._uri; A._uri = lambda fig, **k: buf.setdefault("f", fig) or "x"
        try:
            render()
        except Exception as e:
            print(f"    (skip: {e})")
        finally:
            A._uri = orig
        return buf.get("f")

    def clean(f, size, stem, xlabel=None, rot=False):
        if not f:
            return
        f.suptitle("")
        for ax in f.axes:
            ax.set_title("")
            if rot and len(ax.get_xticklabels()) >= 4:
                for t in ax.get_xticklabels():
                    t.set_rotation(20); t.set_ha("right"); t.set_fontsize(6.0)
            if xlabel is not None:
                ax.set_xlabel(xlabel, fontsize=7)
        while f.texts:
            f.texts[0].remove()
        f.set_size_inches(*size); _save(f, stem)

    clean(cap(lambda: A.fig_retouch_validation()), (3.2, 2.4), "figA2a_speckle", rot=True)
    clean(cap(lambda: A.fig_cross_vendor(cv)), (3.2, 2.4), "figA2b_crossvendor",
          xlabel="peripapillary angle (degrees)")


# ── Appendix — subgroup robustness (C4): decision-flip by stratum, aggregated by reader CLASS (as in Fig 2/3) ──
_A5_CLASSES = ["OCT/generic FM", "ImageNet CNN", "from-scratch CNN", "frontier VLM"]


def _a5_cls(arm):                                                       # reader class, matching the Fig 2/3 legend
    a = arm.replace("_map", "")
    if a.startswith("qwen"):
        return "open VLM"
    if a.startswith("medgemma"):
        return "specialist VLM"
    if a.startswith(("gemini", "openai", "claude")):
        return "frontier VLM"
    if a.startswith(("dinov2", "retfound", "mirage")):
        return "OCT/generic FM"
    if a.startswith(("resnet", "convnext")):
        return "ImageNet CNN"
    return "from-scratch CNN"


def figA5_subgroups():
    """Scanner-shift decision-flip by subgroup, aggregated by reader CLASS (the Fig 2/3 legend classes), not per
    model: (a) B-scan severity, (b) thickness-map severity, (c) thickness-map demographics. Markers are the class
    mean across its models; bars span the min--max across those models. The B-scan over-referral concentrates on
    healthy eyes for the appearance-keyed classes; the map axis stays low and the demographic strata cluster, with no
    group elevated. Built from reports/subgroups.json; no model call."""
    S = json.loads((REPORTS / "subgroups.json").read_text())
    sev_c = {"healthy": OI["green"], "early": OI["orange"], "moderate/advanced": OI["vermillion"]}
    msev_c = {"healthy": OI["green"], "glaucoma early (MD>=-6)": OI["orange"], "glaucoma mod/adv (MD<-6)": OI["vermillion"]}

    def _byclass(section):
        byc = {}
        for arm, rec in S[section].items():
            byc.setdefault(_a5_cls(arm), []).append(rec["groups"])
        classes = [c for c in _A5_CLASSES if c in byc][::-1]           # first class on top
        return byc, classes

    def _frame(ax, classes, xlim):
        yb = np.arange(len(classes))
        ax.set_yticks(yb); ax.set_yticklabels(classes, fontsize=7)
        ax.set_ylim(-0.5, len(classes) - 0.5); ax.set_xlim(*xlim); ax.set_xlabel("decision-flip")
        ax.grid(axis="y", visible=False)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    def _agg_sev(ax, section, cmap, xlim):                             # class mean + model min--max, per severity
        byc, classes = _byclass(section)
        order = list(cmap.keys()); offs = np.linspace(0.22, -0.22, len(order))
        for yi, c in enumerate(classes):
            for k, off in zip(order, offs):
                vals = [g[k]["flip"] for g in byc[c] if k in g]
                if not vals:
                    continue
                ax.plot([min(vals), max(vals)], [yi + off] * 2, color=cmap[k], lw=1.3, alpha=0.55, zorder=2)
                ax.scatter(np.mean(vals), yi + off, color=cmap[k], s=17, edgecolor="white", linewidth=0.4, zorder=3)
        _frame(ax, classes, xlim)
        handles = [Line2D([0], [0], marker="o", color="w", mfc=cmap[k], mec="white", ms=5, label=k) for k in order]
        ax.legend(handles=handles, fontsize=5.6, frameon=False, loc="lower right", handletextpad=0.2, labelspacing=0.25)

    def _agg_demo(ax, xlim):                                           # class mean + model spread, per demographic stratum
        byc, classes = _byclass("map_demographics")
        keys = list(next(iter(S["map_demographics"].values()))["groups"].keys())
        pal = [OI["blue"], OI["skyblue"], OI["green"], OI["orange"], OI["purple"], OI["vermillion"], OI["black"]]
        cmap = {k: pal[i % len(pal)] for i, k in enumerate(keys)}
        offs = np.linspace(0.34, -0.34, len(keys))
        for yi, c in enumerate(classes):
            for k, off in zip(keys, offs):
                vals = [g[k]["flip"] for g in byc[c] if k in g]
                if not vals:
                    continue
                ax.plot([min(vals), max(vals)], [yi + off] * 2, color=cmap[k], lw=1.1, alpha=0.55, zorder=2)
                ax.scatter(np.mean(vals), yi + off, color=cmap[k], s=13, edgecolor="white", linewidth=0.3, zorder=3)
        _frame(ax, classes, xlim)
        handles = [Line2D([0], [0], marker="o", color="w", mfc=cmap[k], mec="white", ms=5, label=k.replace(">=", "≥"))
                   for k in keys]
        ax.legend(handles=handles, fontsize=4.9, frameon=False, loc="lower right", ncol=2, handletextpad=0.2,
                  labelspacing=0.2, columnspacing=0.7)

    fig, axes = plt.subplots(3, 1, figsize=(6.4, 5.6))
    _agg_sev(axes[0], "bscan_severity", sev_c, (-0.02, 0.72))
    _agg_sev(axes[1], "map_severity", msev_c, (-0.02, 0.30))
    _agg_demo(axes[2], (-0.02, 0.30))
    for ax, tag in zip(axes, ("(a) B-scan axis: disease severity", "(b) thickness-map axis: disease severity",
                              "(c) thickness-map axis: demographics")):
        ax.set_title(tag, fontsize=7.6, loc="left", pad=3)
    fig.subplots_adjust(left=0.22, right=0.98, top=0.955, bottom=0.09, hspace=0.42)
    _save(fig, "figA5_subgroups")


def _clean_stale():
    keep = {"fig1_overview", "fig2_stability", "fig3_mechanism", "figA4_grounding",
            "figA1a_bland_altman", "figA1b_sector_loa", "figA1c_dose_response",
            "figA2a_speckle", "figA2b_crossvendor", "figA3_reliability", "figA5_subgroups"}
    for p in FIGDIR.glob("*.p*"):
        if p.stem not in keep:
            p.unlink()


def main():
    FIGDIR.mkdir(parents=True, exist_ok=True)
    app, mp, dev, valid, cv = _load()
    print("rendering submission figures → papers/oct-formatshift-vlm4rwd/figures/")
    fig1_overview()
    fig2_stability(app, mp, dev)
    fig3_mechanism()          # new main Fig 3 (F3 mechanism)
    figA4_grounding(app, mp)   # demoted to the appendix (still generated)
    appendix(valid, cv)
    figA3_reliability()
    figA5_subgroups()          # appendix subgroup-robustness forest (C4)
    _clean_stale()
    n = len(list(FIGDIR.glob("*.pdf")))
    print(f"done — {n} figure PDFs in {FIGDIR}")


if __name__ == "__main__":
    main()
