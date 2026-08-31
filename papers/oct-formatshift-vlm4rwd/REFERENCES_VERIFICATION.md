# Reference verification report (nature-academic-search → nature-ref-verifier)

Every entry was checked against a primary source; no BibTeX was written from memory without grounding.
**web-verified** = title/authors/venue confirmed via a fetched primary source.
**canonical** = widely-known paper, standard/stable metadata, included without a fresh fetch (flagged for a
camera-ready re-check).

A full pre-submission audit (three parallel verification passes + Crossref cross-checks) was run on 2026-08-26.
Fixes applied in that pass are recorded in the **Audit fixes** column.

| Key | Status | Source checked | Audit fixes (2026-08-26) |
|---|---|---|---|
| `warner2011loa` | web-verified | PMC3154907 — PLoS ONE 6(8):e22947, 2011 | reports GLOBAL Cirrus↔Spectralis LoA (bias +1.94, LoA −5.74/+9.62 µm), not per-sector; ±7.68 half-width matches the reported asymmetric bounds; sup/inf profile is mechanism-motivated |
| `ran2019oct` | web-verified | CrossRef 10.1016/S2589-7500(19)30085-8 — Lancet Digit Health 1(4):e172–e182, 2019 (14 authors) | — |
| `maetschke2019` | web-verified | CrossRef 10.1371/journal.pone.0219126 — PLoS ONE 14(7):e0219126, 2019 (raw-OCT CNN AUC 0.94 ≥ RNFL-thickness 0.89) | — |
| `girft2026glaucoma` | web-verified | gettingitrightfirsttime.co.uk — *GIRFT Best Practice for Glaucoma Services*, NHS England, May 2026 (org. author) | — |
| `gedde2021ppp` | web-verified | PubMed/aaojournal — AAO POAG PPP, Ophthalmology 128(1):P71–P150, 2021, doi 10.1016/j.ophtha.2020.10.022 | — |
| `niceng81` | web-verified | nice.org.uk/guidance/ng81 — Glaucoma: diagnosis and management, NICE, 2017 (upd. 2022) | — |
| `jindal2019` | web-verified | PMC / doi 10.1111/opo.12613 — Ophthalmic Physiol Opt 39(3):205–215, 2019 | — |
| `belghith2025crossdomain` | web-verified | PubMed 40833328 — TVST 14(8):29, 2025, doi 10.1167/tvst.14.8.29 | confirmed genuine CycleGAN image-translation (cross-domain) study; claim supported |
| `qin2024lmod` | web-verified | arXiv:2410.01620 — LMOD ophthalmology LVLM benchmark, 2024 | — |
| `alryalat2024chatgpt` | web-verified | Crossref 10.3389/fopht.2024.1387190 — Front Ophthalmol 4:1387190, 2024 | **author fixed**: Musleh, Ahmad M. → **Ayman M.** (Crossref: Ayman Mohammed Musleh) |
| `koh2021wilds` | canonical | ICML 2021 (PMLR 139:5637–5664) — WILDS; arXiv:2012.07421 | — |
| `tsai2022crosscamera` | web-verified | Crossref 10.1155/2022/5779276 — J Diabetes Res 2022:5779276 | **first author + list fixed**: Cheng-Yu Tsai/others → **Meng-Ju Tsai**, Hsieh, Tsai, Chen, Hsieh, Tsai, Chen (7) |
| `chen2024gmaimmbench` | web-verified | arXiv:2408.03361 — GMAI-MMBench, NeurIPS 2024 D&B | — |
| `jalili2025tvst` | web-verified | doi 10.1167/tvst.14.11.31 — TVST 14(11):31, 2025 (LLaVA 0.92 / RETFound 0.91, single Spectralis, no shift/calibration) | — |
| `jalili2024gpt4v` | web-verified | ophthalmologyscience.org S2666-9145(24)00203-3 — Ophthalmol Sci 5(2):100667 | **prose fixed**: uses 3 public *fundus* datasets (ACRIMA/ORIGA/RIM-One), not "single-scanner"; reworded to "grades fundus images, accuracy only" |
| `luo2023harvardgdp` | web-verified | Crossref 10.1109/ICCV51070.2023.01872 — Harvard-GDP, ICCV 2023 | **authors fixed**: 8 → **5** (Luo, Shi, Tian, Elze, Wang; removed Pasquale/Shen/Zebardast); DOI added |
| `gleam2026` | web-verified | Crossref 10.1109/TMI.2026.3692814 — IEEE TMI **45(7):4123–4136, 2026** | **published** (was "in press"): DOI + vol/issue/pages added; authors 8 → **11** (added Hongyan Xu, Ruiting Zhou, Man Tang). n=360 in the paper is our 55/15/30 split of the 1,200-sample set (documented in the datasets table), not GLEAM's 6:2:2 |
| `morano2025mirage` | web-verified | arXiv:2506.08900 (10 authors) | — |
| `liu2024octcubem` | web-verified | arXiv:2408.11227 — OCTCube-M (12 authors) | — |
| `oquab2023dinov2` | web-verified | arXiv:2304.07193 | — |
| `xia2024cares` | web-verified | arXiv:2406.06007 — CARES, NeurIPS D&B 2024 | — |
| `goals2022` | web-verified | arXiv:2207.14447 — GOALS Challenge (OMIA9/MICCAI 2022); Topcon DRI Swept-Source OCT (device confirmed in PDF) | — |
| `driessen2024` | web-verified | TVST 13(12):5, 2024, doi 10.1167/tvst.13.12.5 — device-agnostic ONH-OCT segmenter (nnU-Netv2; Spectralis/Cirrus/Topcon); 12 authors | **prose fixed**: paper does not use the name "Eyened"; all "Eyened-segmented" → "device-agnostically segmented" |
| `diciccio1996bootstrap` | web-verified | Statistical Science 11(3):189–228, 1996, doi 10.1214/ss/1032280214 — DiCiccio & Efron | — |
| `cheung2009scancircle` | web-verified | Crossref 10.1038/eye.2008.258 — Eye 23(6):1436–1441, 2009 (PMID 18806767); scan-circle displacement → sup/inf most affected | — |
| `mansberger2017segerrors` | web-verified | Crossref 10.1016/j.ajo.2016.10.020 — Am J Ophthalmol 174:1–8, 2017; automated RNFL segmentation errors | **prose fixed**: single-device (Spectralis) study; no longer carries "dominate the cross-device variance" — reworded to "concentrate in superior/inferior bundles" |
| `buchser2012` | web-verified | Crossref 10.1167/iovs.11-8432 — IOVS 53(7):3742–3747, 2012; RNFL bias/imprecision across 3 SD-OCT | **now carries** the non-interchangeability / different-RNFL claim (reassigned from `leite2011`) |
| `kumar2022finetuning` | web-verified | arXiv:2202.10054 — ICLR 2022 (Oral) | — |
| `tian2023calibration` | web-verified | ACL Anthology 2023.emnlp-main.330 (8 authors) | — |
| `xiong2024uncertainty` | web-verified | arXiv:2306.13063 — ICLR 2024 (7 authors) | — |
| `lin2022verbalized` | web-verified | arXiv:2205.14334 — *Teaching Models to Express Their Uncertainty in Words*, TMLR 2022 (Lin, Hilton, Evans) | **added** (verbalized-uncertainty origin) |
| `ovadia2019shift` | web-verified | arXiv:1906.02530 — *Can You Trust Your Model's Uncertainty?*, NeurIPS 2019 (9 authors) | **added** (calibration degrades under dataset shift) |
| `groot2024overconfidence` | web-verified | ACL Anthology 2024.trustnlp-1.13 / arXiv:2405.02917 | **sign clarified in prose**: Groot's NCE = accuracy − confidence (negative = overconfident); paper now states it reports the sign-flipped convention (confidence − accuracy; + = overconfident) |
| `li2023llavamed` | web-verified | arXiv:2306.00890 — *LLaVA-Med*, NeurIPS 2023 D&B (Li, Wong, Zhang, Usuyama, Liu, Yang, Naumann, Poon, Gao) | **added** (biomedical instruction-tuned VLM) |
| `rohrbach2018chair` | web-verified | ACL Anthology D18-1437 — EMNLP 2018 (5 authors) | — |
| `li2023pope` | web-verified | ACL Anthology 2023.emnlp-main.20 — POPE (6 authors) | — |
| `zhou2023retfound` | canonical | Nature 622(7981):156–163, 2023 (RETFound) | — |
| `guo2017calibration` | canonical | ICML 2017, PMLR 70:1321–1330 | — |
| `bogunovic2019retouch` | canonical | IEEE TMI 38(8):1858–1874, 2019 (RETOUCH) | — |
| `tham2014global` | canonical | Ophthalmology 121(11):2081–2090, 2014 | — |
| `schmitt1999speckle` | canonical | J Biomed Opt 4(1):95–105, 1999 (doi 10.1117/1.429925) | — |
| `blandaltman1986` | canonical | The Lancet 327(8476):307–310, 1986 | — |
| `hendrycks2019corruptions` | canonical | ICLR 2019 (common corruptions; defines the flip rate we adopt) | — |

**Deleted this audit** (no longer cited): `leite2011` (its non-interchangeability claim was a misattribution —
Leite et al. found the three devices detect glaucoma *equally well*; the claim was reassigned to `buchser2012`);
`zheng2016stability` and `tobin2017domainrand` (dangling after domain-randomization was dropped from the paper).

**Dropped from the outline's long list** (not load-bearing, so not cited rather than padded): Patel 2012,
Cheung–Leung 2008, Mwanza 2010, Taori 2020, Pearl–Bareinboim 2014, Qiu 2022, MedHEval, Med-HALT, MedVH.

All cited keys resolve with zero "undefined citation" warnings at build; no dangling entries remain.
