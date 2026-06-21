"""
Download and extract UrbanSound8K
=================================

UrbanSound8K is a labeled audio dataset of 8732 short urban sounds across
10 classes (car_horn, dog_bark, engine_idling, siren, etc.), pre-sorted
into 10 folds for cross-validation.

Used here as the bootstrap dataset for the audio branch of the fusion model.

Source: https://zenodo.org/records/1203745  (Salamon, Jacoby, Bello — 2014)

Usage:
    .venv/bin/python scripts/download_urbansound8k.py
    .venv/bin/python scripts/download_urbansound8k.py --keep-tar

After completion:
    data/urbansound8k/UrbanSound8K/
        ├── audio/fold{1..10}/*.wav
        └── metadata/UrbanSound8K.csv
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import truststore
truststore.inject_into_ssl()

URL = "https://zenodo.org/records/1203745/files/UrbanSound8K.tar.gz"
DATA_ROOT = Path("data/urbansound8k")
TAR_PATH = DATA_ROOT / "UrbanSound8K.tar.gz"
EXTRACT_DIR = DATA_ROOT / "UrbanSound8K"
METADATA_CSV = EXTRACT_DIR / "metadata" / "UrbanSound8K.csv"

EXPECTED_TOTAL = 8732
EXPECTED_CLASSES = 10


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--keep-tar", action="store_true",
                   help="Don't delete the 6GB tar after extraction")
    p.add_argument("--skip-download", action="store_true",
                   help="Use existing tar (or already-extracted dir)")
    return p.parse_args()


def download(url: str, dest: Path) -> None:
    """Use curl for the heavy lifting — shows progress, handles resume, uses system certs."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    print(f"           → {dest}")
    # -L: follow redirects, --fail: error on HTTP error, -C -: resume if partial
    cmd = ["curl", "-L", "--fail", "-C", "-", "-o", str(dest), url]
    res = subprocess.run(cmd)
    if res.returncode != 0:
        raise RuntimeError(f"curl exited {res.returncode}")
    print(f"Downloaded {dest.stat().st_size / 1e9:.2f} GB")


def extract(tar: Path, into: Path) -> None:
    print(f"\nExtracting {tar} → {into}")
    into.mkdir(parents=True, exist_ok=True)
    res = subprocess.run(["tar", "xzf", str(tar), "-C", str(into)])
    if res.returncode != 0:
        raise RuntimeError(f"tar exited {res.returncode}")


def verify(extracted_root: Path) -> int:
    print(f"\nVerifying contents...")

    if not METADATA_CSV.exists():
        print(f"  FAIL: metadata CSV not found at {METADATA_CSV}", file=sys.stderr)
        return 1

    with METADATA_CSV.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    total = len(rows)
    class_counts = Counter(r["class"] for r in rows)
    fold_counts  = Counter(r["fold"] for r in rows)

    print(f"  Total samples:    {total}")
    print(f"  Distinct classes: {len(class_counts)}")
    print()
    print(f"  Per-class counts:")
    for cls, n in sorted(class_counts.items()):
        print(f"    {cls:<22} {n}")
    print()
    print(f"  Per-fold counts:")
    for fold, n in sorted(fold_counts.items(), key=lambda x: int(x[0])):
        print(f"    fold {fold:<3}             {n}")

    # Spot-check a few WAV files actually exist on disk
    misses = 0
    for r in rows[:50]:
        wav = extracted_root / "audio" / f"fold{r['fold']}" / r["slice_file_name"]
        if not wav.exists():
            misses += 1
    print(f"\n  First-50 WAV existence check: {50 - misses}/50 present")

    ok = total == EXPECTED_TOTAL and len(class_counts) == EXPECTED_CLASSES and misses == 0
    if ok:
        print(f"\n  OK — dataset looks complete and well-formed.")
        return 0
    else:
        print(f"\n  WARN — counts deviate from expected "
              f"(expected total={EXPECTED_TOTAL}, classes={EXPECTED_CLASSES})", file=sys.stderr)
        return 1


def main() -> int:
    args = parse_args()

    already_extracted = METADATA_CSV.exists()
    if not already_extracted:
        if not args.skip_download:
            download(URL, TAR_PATH)
        else:
            if not TAR_PATH.exists():
                print(f"--skip-download but {TAR_PATH} not found", file=sys.stderr)
                return 2
        extract(TAR_PATH, DATA_ROOT)

    rc = verify(EXTRACT_DIR)

    # Reclaim ~6GB if extraction succeeded
    if rc == 0 and not args.keep_tar and TAR_PATH.exists():
        size = TAR_PATH.stat().st_size / 1e9
        TAR_PATH.unlink()
        print(f"\n  Removed {TAR_PATH} (reclaimed {size:.2f} GB). Use --keep-tar to retain.")

    return rc


if __name__ == "__main__":
    sys.exit(main())
