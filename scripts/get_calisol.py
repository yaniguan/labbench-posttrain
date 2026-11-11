"""Download the official CALiSol-23 electrolyte conductivity atlas (`13,825` measurements + SMILES).

Source:
- De Breuck et al. (2024), DTU Data DOI: `10.11583/DTU.c.6929599.v1`
- Mirror: `https://github.com/Pele0599/CALiSol-23`
"""
from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "calisol"

FILES = {
    "calisol23.csv": "https://raw.githubusercontent.com/Pele0599/CALiSol-23/main/CALiSol-23%20Dataset.csv",
    "calisol_smiles.csv": "https://raw.githubusercontent.com/Pele0599/CALiSol-23/main/calisolsmile.csv",
}


def download_calisol(out_dir: str | Path = DEFAULT_OUT_DIR, force: bool = False) -> dict[str, Path]:
    target_dir = Path(out_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for fname, url in FILES.items():
        dest = target_dir / fname
        if force or not dest.exists():
            urllib.request.urlretrieve(url, dest)  # noqa: S310
        paths[fname] = dest
    return paths


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fetch CALiSol-23 CSV and SMILES table.")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    out = download_calisol(args.out_dir, force=args.force)
    for k, p in out.items():
        print(f"{k}: {p} ({p.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
