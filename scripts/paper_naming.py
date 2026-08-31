"""Single source of truth for reader display names across the OCT-FormatShift paper.

Every table (LaTeX) and figure (matplotlib) derives its reader label from ``disp(key, fmt)`` so the names match
Table 1 exactly. Probe suffix uses a middle dot (DINOv2·FT); VLMs use the short model name plus a few-shot/
zero-shot marker (Gemini 3.6 fs). ``fmt='latex'`` emits the Table-1 form (\\,$\\cdot$\\, and \\textit{fs}, Claude
Opus~5, SmallCNN (98k)); ``fmt='fig'`` emits a matplotlib-safe unicode form (·, plain fs, Claude Opus 5).
"""
from __future__ import annotations

# raw contender key (with any "_map" stripped) -> (base name, probe suffix or None)
NONVLM = {
    "dinov2_ft": ("DINOv2", "FT"), "dinov2_reg": ("DINOv2", "LP"),
    "retfound_ft": ("RETFound", "FT"), "retfound": ("RETFound", "LP"),
    "mirage_ft": ("MIRAGE", "FT"), "mirage": ("MIRAGE", "LP"),
    "resnet50_ft": ("ResNet-50", "FT"), "resnet50": ("ResNet-50", "LP"),
    "convnext_tiny_ft": ("ConvNeXt-T", "FT"), "convnext_base_ft": ("ConvNeXt-B", "FT"),
    "convnext_large_ft": ("ConvNeXt-L", "FT"), "convnext_tiny": ("ConvNeXt-T", "LP"),
    "convnext_base": ("ConvNeXt-B", "LP"), "convnext_large": ("ConvNeXt-L", "LP"),
    "supervised": ("SmallCNN", None),
}


def _vlm_base(k: str) -> str | None:
    if "gemini" in k:
        if "3p6" in k:
            return "Gemini 3.6 Flash"
        if "2p5_pro" in k:
            return "Gemini 2.5 Pro"
        if "2p5_flash" in k:
            return "Gemini 2.5 Flash"
        return "Gemini"
    if "openai" in k or k.startswith("gpt"):
        return "GPT-5.6 Sol"
    if "claude" in k:
        return "Claude Opus 5"
    if "qwen" in k:
        return "Qwen3-VL-8B"
    if "medgemma" in k:
        return "MedGemma-1.5-4B"
    return None


def disp(key: str, fmt: str = "fig") -> str:
    """Canonical display name for a contender key. fmt in {'fig','latex'}."""
    latex = fmt == "latex"
    base = _vlm_base(key)
    if base is not None:
        marker = "fs" if "fewshot" in key else ("zs" if "zeroshot" in key else "")
        if latex:
            name = base.replace("Claude Opus 5", "Claude Opus~5")
            return name + (f"\\,\\textit{{{marker}}}" if marker else "")
        return base + (f" {marker}" if marker else "")
    kk = key.replace("_map", "")
    if kk in NONVLM:
        name, suf = NONVLM[kk]
        if name == "SmallCNN":
            return "SmallCNN (98k)" if latex else "SmallCNN"
        if suf is None:
            return name
        return name + (f"\\,$\\cdot$\\,{suf}" if latex else f"·{suf}")
    return kk.replace("_", "\\_") if latex else kk
