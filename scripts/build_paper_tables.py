#!/usr/bin/env python
"""Emit the full per-axis appendix tables (booktabs) straight from the committed panels — no transcription.

    .venv/bin/python scripts/build_paper_tables.py
writes papers/oct-formatshift-vlm4rwd/appendix_tables.tex
"""
from __future__ import annotations
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import build_oct_format_artifact as A
from paper_naming import disp

OUT = ROOT / "papers" / "oct-formatshift-vlm4rwd" / "appendix_tables.tex"
REPORTS = ROOT / "reports"

LABEL = {"vlm_closed": "frontier VLM", "vlm_open": "open VLM", "vlm_spec": "specialist VLM",
         "fm": "FM", "cnn": "ImageNet CNN", "scratch": "scratch CNN", "floor": "floor/oracle"}


def _cls(m):
    if m.startswith(("gemini", "openai", "claude")): return "vlm_closed"
    if m.startswith("qwen3vl"): return "vlm_open"
    if m.startswith("medgemma"): return "vlm_spec"
    if m.startswith(("retfound", "dinov2", "mirage")): return "fm"
    if m.startswith(("resnet", "convnext")): return "cnn"
    if m in ("supervised", "fullres_cnn", "fullres_cnn_dr") or "SmallCNN" in m: return "scratch"
    return "floor"


def _name(m):
    if A._is_vlm(m):
        lab = A._vlm_label(m)
        lab = lab[0] if isinstance(lab, (tuple, list)) else lab
        return str(lab).replace("&", "\\&")
    return m.replace("_", "\\_")


def _rows(panel, nce, calib):
    out = []
    order = {"fm": 0, "cnn": 1, "scratch": 2, "floor": 3, "vlm_closed": 4, "vlm_open": 5, "vlm_spec": 6}
    for m in sorted(panel, key=lambda k: (order[_cls(k)], -panel[k]["per_level"]["1.0"]["auc_clean"])):
        r = panel[m]["per_level"]["1.0"]; ci = r.get("ci", {})
        def cif(key):
            c = ci.get(key)
            return f" \\ci{{{c[0]:.2f}}}{{{c[1]:.2f}}}" if isinstance(c, (list, tuple)) else ""
        spec = r.get("spec_realized_shift")
        spec_s = f"{spec:.2f}" if isinstance(spec, (int, float)) else "--"
        nce_v = nce.get(m)
        nce_s = f"{nce_v:+.3f}" if isinstance(nce_v, (int, float)) else "--"
        cm = calib.get(m, {}).get("clean", {})              # adaptive-ECE + Brier on the same clean probs
        aece = cm.get("ece_adaptive"); brier = cm.get("brier")
        aece_s = f"{aece:.3f}" if isinstance(aece, (int, float)) else "--"
        brier_s = f"{brier:.3f}" if isinstance(brier, (int, float)) else "--"
        out.append(f"    {disp(m, 'latex')} & {LABEL[_cls(m)]} & {r['auc_clean']:.3f}{cif('auc_clean')} & "
                   f"{r['referral_flip']:.3f}{cif('referral_flip')} & {r['delta_sens_at_spec']:+.3f}{cif('delta_sens_at_spec')} & "
                   f"{spec_s} & {r['ece_clean']:.3f} & {aece_s} & {brier_s} & {nce_s} & {r['delta_auc']:+.3f} \\\\")
    return "\n".join(out)


def _table(panel, title, sub, n, nce, calib):
    return (f"\\begin{{table}}[h]\n  \\centering\n  \\caption{{{title}. {sub} ($n{{=}}{n}$ eyes). "
            "spec@shift = realized specificity under the shift at the 90\\%-specificity operating point (a low value "
            "signals decision collapse); ECE, aECE and Brier have no bootstrap CI in the released summaries. "
            "ECE is the positive-class calibration error (10 equal-width bins); aECE is the adaptive (equal-mass) "
            "ECE and Brier is the Brier score, a strictly proper score with no binning, so together they show the "
            "calibration reads are not a fixed-bin artifact. NCE $=$ net calibration error, a signed calibration "
            "error \\citep{groot2024overconfidence} (mean predicted-class confidence minus accuracy; $+$ $=$ "
            "overconfident). All calibration reads are on the clean eyes.}\n"
            f"  \\small\n  \\setlength{{\\tabcolsep}}{{3pt}}\n  \\resizebox{{\\linewidth}}{{!}}{{%\n"
            f"  \\begin{{tabular}}{{llccccccccc}}\n    \\toprule\n"
            "    Reader & Class & AUC \\ci{95\\%}{CI} & Flip \\ci{95\\%}{CI} & $\\Delta$sens \\ci{95\\%}{CI} & "
            "spec@shift & ECE & aECE & Brier & NCE & $\\Delta$AUC \\\\\n    \\midrule\n"
            f"{_rows(panel, nce, calib)}\n    \\bottomrule\n  \\end{{tabular}}}}\n\\end{{table}}\n")


# reader row order (by class) for the subgroup tables, as canonical plain display names (== disp(key,'fig'))
_SUB_ORDER = ["DINOv2·FT", "DINOv2·LP", "RETFound·FT", "RETFound·LP", "MIRAGE·FT", "MIRAGE·LP",
              "ResNet-50·FT", "ResNet-50·LP", "ConvNeXt-B·LP", "ConvNeXt-L·LP", "SmallCNN",
              "Gemini 3.6 Flash fs", "GPT-5.6 Sol fs", "Claude Opus 5 fs"]


def _subgroup_tables():
    """Emit tab:sev (severity, both axes) and tab:demo (map demographics) for all capable readers, straight from
    reports/subgroups.json (point estimates; 95% CIs are in Fig.~\\ref{fig:subgroups}). Names via the shared
    canonical scheme (disp), keyed by the plain form so B-scan and map rows align to one reader."""
    S = json.loads((REPORTS / "subgroups.json").read_text())

    def by_reader(section):                                   # plain-name -> (latex-name, groups)
        return {disp(k, "fig"): (disp(k, "latex"), v["groups"]) for k, v in S[section].items()}

    bs, ms, dm = by_reader("bscan_severity"), by_reader("map_severity"), by_reader("map_demographics")

    def cell(rec, key):
        return f"{rec[1][key]['flip']:.2f}" if rec and key in rec[1] else "--"

    def latexname(*recs):
        return next(r[0] for r in recs if r)

    order_cls = {"fm": 0, "cnn": 1, "scratch": 2, "floor": 3, "vlm_closed": 4, "vlm_open": 5, "vlm_spec": 6}

    def _order(*sections):                                   # all readers across sections, by class then name
        reps = {}
        for sec in sections:
            for k in S[sec]:
                reps.setdefault(disp(k, "fig"), k)
        return sorted(reps, key=lambda p: (order_cls[_cls(reps[p])], p))

    bkeys = ["healthy", "early", "moderate/advanced"]
    mkeys = ["healthy", "glaucoma early (MD>=-6)", "glaucoma mod/adv (MD<-6)"]
    sev_rows = []
    for lab in _order("bscan_severity", "map_severity"):
        b, m = bs.get(lab), ms.get(lab)
        if b is None and m is None:
            continue
        cells = [cell(b, k) for k in bkeys] + [cell(m, k) for k in mkeys]
        sev_rows.append(f"    {latexname(b, m)} & " + " & ".join(cells) + " \\\\")
    sev = ("\\begin{table}[tbp]\n  \\centering\n  \\caption{\\textbf{Decision-flip by disease severity}, both axes, "
           "all configurations (point estimates; the per-reader eye-level bootstrap 95\\% CIs are wide and overlapping "
           "across grades, and the class-aggregated means and model spread are in Fig.~\\ref{fig:subgroups}a,b). "
           "Severity: B-scan from the GLEAM 4-class label "
           "(healthy / early / moderate-advanced); map from Harvard-GDP mean deviation (healthy / MD$\\ge-6$ / "
           "MD$<-6$). ``--'' = reader not run on that axis.}\n  \\label{tab:sev}\n  \\small\n"
           "  \\renewcommand{\\arraystretch}{0.9}\n  \\setlength{\\tabcolsep}{5pt}\n  \\begin{tabular}{lcccccc}\n    \\toprule\n"
           "    & \\multicolumn{3}{c}{B-scan axis} & \\multicolumn{3}{c}{Thickness-map axis} \\\\\n"
           "    \\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n"
           "    Reader & healthy & early & mod/adv & healthy & early & mod/adv \\\\\n    \\midrule\n"
           + "\n".join(sev_rows) + "\n    \\bottomrule\n  \\end{tabular}\n\\end{table}\n")

    demo_order = list(next(iter(S["map_demographics"].values()))["groups"].keys())

    def _hdr(k):
        if k == "female":
            return "F"
        if k == "male":
            return "M"
        if k.startswith("age<"):
            return "$<$" + k[4:]
        if k.startswith("age>="):
            return "$\\ge$" + k[5:]
        return k
    demo_rows = []
    for lab in _order("map_demographics"):
        d = dm.get(lab)
        if d is None:
            continue
        demo_rows.append(f"    {latexname(d)} & " + " & ".join(cell(d, k) for k in demo_order) + " \\\\")
    demo = ("\\begin{table}[tbp]\n  \\centering\n  \\caption{\\textbf{Decision-flip by demographics} (thickness-map "
            "axis, Harvard-GDP; all configurations, point estimates, wide overlapping CIs; class-aggregated view in "
            "Fig.~\\ref{fig:subgroups}c). No group is consistently elevated. GLEAM (B-scan) carries no "
            "demographics.}\n  \\label{tab:demo}\n  \\small\n"
            "  \\renewcommand{\\arraystretch}{0.9}\n  \\setlength{\\tabcolsep}{5pt}\n  \\begin{tabular}{lccccccc}\n    \\toprule\n"
            "    & \\multicolumn{2}{c}{gender} & \\multicolumn{3}{c}{race} & \\multicolumn{2}{c}{age} \\\\\n"
            "    \\cmidrule(lr){2-3}\\cmidrule(lr){4-6}\\cmidrule(lr){7-8}\n"
            "    Reader & " + " & ".join(_hdr(k) for k in demo_order) + " \\\\\n    \\midrule\n"
            + "\n".join(demo_rows) + "\n    \\bottomrule\n  \\end{tabular}\n\\end{table}\n")

    outp = ROOT / "papers" / "oct-formatshift-vlm4rwd" / "subgroup_tables.tex"
    outp.write_text("% auto-generated by scripts/build_paper_tables.py — do not edit by hand\n" + sev + "\n" + demo)
    print(f"wrote {outp}  ({len(sev_rows)} severity rows, {len(demo_rows)} demographic rows)")


def main():
    _subgroup_tables()
    app = json.load(open(REPORTS / "oct_appearance_axis_panel.json"))["panel"]
    mp = json.load(open(REPORTS / "oct_map_fm_panel.json"))["panel"]
    nce = json.load(open(REPORTS / "nce_all.json"))
    calib = json.load(open(REPORTS / "calibration_metrics.json"))["axes"]
    na = app[next(iter(app))]["per_level"]["1.0"]["n"]; nm = mp[next(iter(mp))]["per_level"]["1.0"]["n"]
    txt = ("% auto-generated by scripts/build_paper_tables.py — do not edit by hand\n"
           + _table(app, f"\\textbf{{Full B-scan (appearance) axis, all {len(app)} configurations}}",
                    "Spectralis\\,$\\rightarrow$\\,Topcon speckle shift", na, nce["app"], calib["bscan"])
           + "\n" + _table(mp, f"\\textbf{{Full thickness-map (color-map) axis, all {len(mp)} configurations}}",
                           "Cirrus\\,$\\rightarrow$\\,Spectralis offset; OCT-specific B-scan FMs are not run on map renders",
                           nm, nce["mp"], calib["map"]))
    OUT.write_text(txt)
    print(f"wrote {OUT}  ({len(app)} + {len(mp)} configurations)")


if __name__ == "__main__":
    main()
