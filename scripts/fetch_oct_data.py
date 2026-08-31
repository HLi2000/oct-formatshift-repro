#!/usr/bin/env python
"""Fetch the OCT-FormatShift large binaries (FM/CNN contender npz + raw datasets) from the private
`data-cache` GitHub Release, verify them against reports/oct_format_data_manifest.json, and place/extract
them into data/derived/ + data/raw/.

    GH_TOKEN=... .venv/bin/python scripts/fetch_oct_data.py            # all assets
    GH_TOKEN=... .venv/bin/python scripts/fetch_oct_data.py oct_contender_npz.tar   # just the npz (Fig 3/A3)

The release is PRIVATE, so a GH_TOKEN with read access is required (env var or .env). Committed VLM npz +
reports/*.json already reproduce every NUMBER and the VLM figure curves; this script only adds the FM/CNN npz
(needed to regenerate Fig 3 + Fig A3) and the raw datasets (Fig 1, A1a, A2a + full panel regeneration).
"""
from __future__ import annotations
import hashlib, json, os, subprocess, sys, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "reports" / "oct_format_data_manifest.json"
API = "https://api.github.com"


def _token() -> str:
    tok = os.environ.get("GH_TOKEN")
    if not tok and (ROOT / ".env").exists():
        for line in (ROOT / ".env").read_text().splitlines():
            if line.startswith("GH_TOKEN="):
                tok = line.split("=", 1)[1].strip().strip('"').strip("'")
                break
    if not tok:
        sys.exit("GH_TOKEN not set (env or .env) — required for the private data-cache release.")
    return tok


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _release_assets(repo: str, tag: str, tok: str) -> dict:
    req = urllib.request.Request(f"{API}/repos/{repo}/releases/tags/{tag}",
                                 headers={"Authorization": f"token {tok}", "Accept": "application/vnd.github+json"})
    rel = json.load(urllib.request.urlopen(req, timeout=30))
    return {a["name"]: a for a in rel["assets"]}


def _download(repo: str, asset_id: int, dest: Path, tok: str, expected_size: int | None = None, attempts: int = 5):
    """Download an asset via curl, retrying until the full file lands. We shell out to curl (universal on
    macOS/Linux) rather than urllib: curl's HTTP/1.1 + --retry handles the GitHub->S3 redirect and flaky
    connections robustly, whereas a plain urllib stream can silently truncate large transfers. Each outer
    attempt re-requests the API endpoint (fresh redirect URL) and the byte count is re-checked."""
    url = f"{API}/repos/{repo}/releases/assets/{asset_id}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    for i in range(1, attempts + 1):
        rc = subprocess.run(
            ["curl", "-fL", "--http1.1", "--retry", "5", "--retry-all-errors", "--retry-delay", "2",
             "--max-time", "3600", "-s", "-H", f"Authorization: token {tok}",
             "-H", "Accept: application/octet-stream", "-o", str(dest), url],
        ).returncode
        got = dest.stat().st_size if dest.exists() else 0
        if rc == 0 and (expected_size is None or got == expected_size):
            return True
        print(f"    attempt {i}/{attempts}: curl rc={rc}, {got}/{expected_size} bytes, retrying")
    return False


def main():
    man = json.loads(MANIFEST.read_text())
    repo, tag = man["release"]["repo"], man["release"]["tag"]
    want = set(sys.argv[1:]) or set(man["assets"])
    tok = _token()
    assets = _release_assets(repo, tag, tok)
    staging = ROOT / "data" / "_dl"
    for name in want:
        spec = man["assets"][name]
        if name not in assets:
            print(f"[!] {name} not in release {tag} — skipping"); continue
        dest = staging / name
        print(f"[fetch] {name} ({spec['size']/1e6:.1f} MB)...")
        if not _download(repo, assets[name]["id"], dest, tok, spec["size"]):
            sys.exit(f"[x] {name}: could not fetch the full file after retries "
                     f"({dest.stat().st_size if dest.exists() else 0}/{spec['size']} bytes)")
        # verify
        if dest.stat().st_size != spec["size"]:
            sys.exit(f"[x] {name}: size {dest.stat().st_size} != manifest {spec['size']}")
        if spec.get("sha256"):
            got = _sha256(dest)
            if got != spec["sha256"]:
                sys.exit(f"[x] {name}: sha256 mismatch\n   got  {got}\n   want {spec['sha256']}")
            print(f"    sha256 OK")
        else:
            print(f"    size OK (no sha256 in manifest)")
    print("\nExtraction (run these; see each asset's 'extract'/'reassemble' in the manifest):")
    print(f"  tar -xf {staging}/oct_contender_npz.tar        # -> data/derived[/map_fm]/*.npz")
    print(f"  unzip -o {staging}/gleam.zip -d data/raw/gleam  ;  unzip -o {staging}/goals.zip -d data/raw/goals")
    print(f"  cat {staging}/RETOUCH_part_aa {staging}/RETOUCH_part_ab > data/raw/retouch/RETOUCH.tar")
    print("(Downloads staged under data/_dl/ — gitignored. Delete after extracting.)")


if __name__ == "__main__":
    main()
