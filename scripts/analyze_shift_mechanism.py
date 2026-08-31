#!/usr/bin/env python
"""F3 mechanism: does the scanner shift SLIDE the score distribution past the fixed threshold (preserving ranking, so
AUC holds while the decision flips), or does it destabilise individual eyes at random?

Two layers, both READ-ONLY (no model re-queried):
  1. SYSTEMATIC (all arms, both axes): for every arm we take the paper's operating-point decision-flip and ΔAUC
     (from the committed panels) and join them with the per-eye distribution-slide magnitude mean Δp (Δp = shifted −
     clean, computed from committed per-eye probs). Across arms we test whether flip tracks the slide, not ΔAUC, and
     which reader classes slide toward "refer" (over-referral).
  2. EXEMPLARS (three B-scan readers): the full per-eye picture (mean/sd Δp, rank correlation, up-fraction, and the
     directionality of the operating-point crossings) that illustrates the coherent slide.

VLM/fine-tuned/supervised arms store clean_prob + perturbed_1.0 directly; frozen-probe FM/CNN arms store cached
features, so we reproduce their probabilities with the same frozen probe used elsewhere (fm_contender._probe on the
seed-0 split), mirroring build_paper_submission_figures._reliability_probs.

Output: reports/shift_mechanism.json .
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")  # sklearn CV FutureWarnings
from src.glaucoma_triage.oct_format import fm_contender as FM  # noqa: E402

DERIVED = ROOT / "data" / "derived"
OUT = ROOT / "reports" / "shift_mechanism.json"

# axis -> (panel json, derived npz dir)
AXES = {
    "bscan": (ROOT / "reports" / "oct_appearance_axis_panel.json", DERIVED),
    "map": (ROOT / "reports" / "oct_map_fm_panel.json", DERIVED / "map_fm"),
}
# three B-scan exemplars carried in the per-eye mechanism narrative (the ΔAUC-small-but-flips cases).
EXEMPLARS = ["mirage", "resnet50_ft", "retfound"]


def _classify(m: str) -> str:
    if m.startswith(("gemini", "openai", "claude")):
        return "frontier VLM"
    if m.startswith("qwen"):
        return "open VLM"
    if m.startswith("medgemma"):
        return "specialist VLM"
    if m.startswith(("retfound", "mirage")):
        return "OCT FM"
    if m.startswith("dinov2"):
        return "generic FM"
    if m.startswith(("resnet", "convnext")):
        return "ImageNet CNN"
    if m in ("supervised", "fullres_cnn", "fullres_cnn_dr"):
        return "from-scratch CNN"
    return "other"


def _panel(path: Path) -> dict:
    """arm -> per-level-1.0 record (op_threshold, referral_flip, delta_auc, spec_realized_shift, auc_clean)."""
    p = json.loads(path.read_text())["panel"]
    out = {}
    for k, v in p.items():
        pl = v.get("per_level", {}).get("1.0")
        if pl:
            out[k] = pl
    return out


def _clean_shift(npz: Path):
    """Return (clean_prob, shift_prob, labels) on the eval eyes for one arm, or None if fields are missing."""
    d = np.load(npz, allow_pickle=True)
    y = np.asarray(d["labels"])
    if "clean_prob" in d:  # VLM / fine-tuned / supervised: probs stored directly on the eval eyes
        if "perturbed_1.0" not in d:
            return None
        return np.asarray(d["clean_prob"], float), np.asarray(d["perturbed_1.0"], float), y.astype(float)
    if not all(k in d for k in ("tr", "te", "clean", "perturbed_1.0")):
        return None  # e.g. eyened (no split)
    tr, te = np.asarray(d["tr"]), np.asarray(d["te"])
    X, Xp = np.asarray(d["clean"], float), np.asarray(d["perturbed_1.0"], float)
    ytr = y[tr]
    pipe = FM._probe(int(ytr.sum()), int((ytr == 0).sum())).fit(X[tr], ytr)
    return pipe.predict_proba(X[te])[:, 1], pipe.predict_proba(Xp[te])[:, 1], y[te].astype(float)


def _mechanism(clean, shift, y, thr: float) -> dict:
    """Per-eye slide record at decision threshold ``thr`` (the arm's operating point)."""
    dp = shift - clean
    call_c, call_s = clean > thr, shift > thr
    flip = call_c != call_s
    return {
        "n": int(y.size),
        "op_threshold": float(thr),
        "mean_dp": float(dp.mean()),
        "sd_dp": float(dp.std()),
        "spearman_clean_shift": float(spearmanr(clean, shift).statistic),
        "frac_up": float((dp > 0).mean()),
        "flip_rate": float(flip.mean()),
        "flips_total": int(flip.sum()),
        "flips_up_cross": int((call_s & ~call_c).sum()),    # negatives crossing up to "refer"
        "flips_down_cross": int((~call_s & call_c).sum()),  # crossing down to "not refer"
        "mean_dp_negatives": float(dp[y == 0].mean()) if (y == 0).any() else float("nan"),
        "mean_dp_positives": float(dp[y == 1].mean()) if (y == 1).any() else float("nan"),
    }


def _degenerate(clean) -> bool:
    return len(set(np.round(clean, 4))) < 3  # near-constant hedgers: exclude from correlation stats


def _systematic(panel: dict, npz_dir: Path) -> dict:
    arms = {}
    for arm, pl in panel.items():
        f = npz_dir / f"{arm}_contender.npz"
        if not f.exists():
            continue
        r = _clean_shift(f)
        if r is None:
            continue
        c, s, y = r
        dp = s - c
        arms[arm] = {
            "cls": _classify(arm),
            "flip": float(pl["referral_flip"]),          # paper operating-point flip
            "delta_auc": float(pl["delta_auc"]),
            "spec_shift": float(pl["spec_realized_shift"]),
            "mean_dp": float(dp.mean()),                  # net (directional) slide
            "mean_abs_dp": float(np.abs(dp).mean()),       # total per-eye movement (mode-agnostic)
            "frac_up": float((dp > 0).mean()),
            "flip_up": float(np.mean((c <= 0.5) & (s > 0.5))),   # newly refer at 0.5 (over-referral); Fig 3c up-bar
            "flip_dn": float(np.mean((c > 0.5) & (s <= 0.5))),   # newly do-not-refer at 0.5; Fig 3c down-bar
            "degenerate": bool(_degenerate(c)),
        }
    keep = [a for a in arms.values() if not a["degenerate"]]
    flip = np.array([a["flip"] for a in keep])
    slide = np.array([abs(a["mean_dp"]) for a in keep])
    total = np.array([a["mean_abs_dp"] for a in keep])
    dauc = np.array([abs(a["delta_auc"]) for a in keep])
    by_cls: dict = {}
    for a in arms.values():
        by_cls.setdefault(a["cls"], []).append(a["mean_dp"])
    summary = {
        "n_arms": len(arms),
        "n_nondegenerate": len(keep),
        "abs_delta_auc_max": float(max(abs(a["delta_auc"]) for a in arms.values())),
        "flip_min": float(min(a["flip"] for a in arms.values())),
        "flip_max": float(max(a["flip"] for a in arms.values())),
        "spec_shift_min": float(min(a["spec_shift"] for a in arms.values())),
        "spec_shift_max": float(max(a["spec_shift"] for a in arms.values())),
        "rho_flip_vs_slide": float(spearmanr(slide, flip).statistic),
        "rho_flip_vs_total_movement": float(spearmanr(total, flip).statistic),
        "rho_flip_vs_delta_auc": float(spearmanr(dauc, flip).statistic),
        "arms_slide_up": int(sum(a["mean_dp"] > 0 for a in arms.values())),
        "by_class_mean_dp": {
            k: {"mean_dp": float(np.mean(v)), "n": len(v), "up": int(sum(x > 0 for x in v))}
            for k, v in sorted(by_cls.items(), key=lambda kv: -np.mean(kv[1]))
        },
    }
    return {"arms": arms, "summary": summary}


def main() -> None:
    out: dict = {
        "_meta": {
            "source": "committed per-eye clean/perturbed_1.0 (probs for VLM/FT/supervised arms; frozen-probe on "
                      "features for FM/CNN); flip and delta_auc are the paper's operating-point values from the panels",
            "reading": "flip tracks |mean_dp| (the coherent slide) not |delta_auc|; high spearman_clean_shift + "
                       "one-directional crossings => the shift slides the distribution past the fixed threshold "
                       "(ranking preserved, so AUC holds), not per-eye rank noise",
        },
        "systematic": {},
        "exemplars": {},
    }
    for axis, (panel_path, npz_dir) in AXES.items():
        panel = _panel(panel_path)
        sysrec = _systematic(panel, npz_dir)
        out["systematic"][axis] = sysrec
        s = sysrec["summary"]
        print(f"[{axis}] {s['n_arms']} arms | |dAUC|max={s['abs_delta_auc_max']:.3f} flip {s['flip_min']:.2f}-"
              f"{s['flip_max']:.2f} | rho(flip,slide)={s['rho_flip_vs_slide']:.2f} rho(flip,dAUC)="
              f"{s['rho_flip_vs_delta_auc']:.2f} | up {s['arms_slide_up']}/{s['n_arms']}")

    # exemplars: full per-eye mechanism at the operating point (B-scan axis)
    bscan_panel = _panel(AXES["bscan"][0])
    for a in EXEMPLARS:
        f = DERIVED / f"{a}_contender.npz"
        if not f.exists() or a not in bscan_panel:
            print(f"[skip exemplar] {a}")
            continue
        c, s, y = _clean_shift(f)
        # referral_flip (and the crossing directionality) is defined at proba>0.5, per degradation.paired_degradation;
        # the 90%-spec operating point governs Δsens@spec/spec@shift only.
        rec = _mechanism(c, s, y, 0.5)
        rec["referral_flip_panel"] = float(bscan_panel[a]["referral_flip"])
        rec["delta_auc_panel"] = float(bscan_panel[a]["delta_auc"])
        rec["spec_shift_panel"] = float(bscan_panel[a]["spec_realized_shift"])
        out["exemplars"][a] = rec
        print(f"  {a:14s} meanΔp={rec['mean_dp']:+.3f} rho={rec['spearman_clean_shift']:.2f} up={rec['frac_up']*100:.0f}% "
              f"flips={rec['flips_total']} ({rec['flips_up_cross']}up/{rec['flips_down_cross']}down) "
              f"panel-flip={rec['referral_flip_panel']:.2f} dAUC={rec['delta_auc_panel']:+.3f}")
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
