"""
Fetch YouTube thumbnails for VGGSound samples
==============================================

Samples N videos per target class from the VGGSound CSV, fetches the YouTube
thumbnail for each (no video download), and discards deleted-video placeholders.

Each kept thumbnail is saved to:
    data/vggsound/thumbnails/{COCO_class}/{youtube_id}.jpg

A manifest with the source VGGSound class and target COCO class is written to:
    data/vggsound/manifest.json

Usage:
    .venv/bin/python scripts/fetch_vggsound_thumbnails.py
    .venv/bin/python scripts/fetch_vggsound_thumbnails.py --per-class 30
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import sys
import time
from pathlib import Path

import ssl
import truststore
truststore.inject_into_ssl()  # Use OS-native cert store (handles corporate TLS proxies)

import requests
from PIL import Image

# VGGSound class -> COCO class we'll compare against
# (the COCO class is what YOLOv8 produces; the VGGSound class is the ground truth)
TARGET_CLASSES = {
    "car passing by":         "car",
    "driving motorcycle":     "motorcycle",
    "dog barking":            "dog",
    "cat meowing":            "cat",
    "bird chirping, tweeting": "bird",
}

CSV_PATH = Path("data/vggsound/vggsound.csv")
OUT_ROOT = Path("data/vggsound/thumbnails")
MANIFEST_PATH = Path("data/vggsound/manifest.json")

# When a YouTube video is deleted, img.youtube.com returns a gray
# placeholder rather than 404. The placeholder is small and a specific size.
# We detect by combining: file size < 3KB AND image dimensions <= 240x180.
MIN_PLAUSIBLE_BYTES = 3000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--per-class", type=int, default=30,
                   help="How many valid thumbnails to keep per class")
    p.add_argument("--oversample", type=int, default=2,
                   help="Multiplier on per-class to account for link rot")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for reproducible sampling")
    p.add_argument("--quality", default="hqdefault",
                   choices=["hqdefault", "sddefault", "maxresdefault", "0"],
                   help="YouTube thumbnail resolution variant")
    return p.parse_args()


def thumb_url(yt_id: str, quality: str) -> str:
    return f"https://img.youtube.com/vi/{yt_id}/{quality}.jpg"


def is_placeholder(content: bytes) -> bool:
    """True if the image is YouTube's deleted-video gray placeholder."""
    if len(content) < MIN_PLAUSIBLE_BYTES:
        return True
    try:
        with Image.open(io.BytesIO(content)) as im:
            w, h = im.size
        return w <= 240 and h <= 180
    except Exception:
        return True


def fetch_one(yt_id: str, quality: str, session: requests.Session, timeout: float = 10) -> bytes | None:
    try:
        r = session.get(thumb_url(yt_id, quality), timeout=timeout)
    except requests.RequestException:
        return None
    if r.status_code != 200:
        return None
    if is_placeholder(r.content):
        return None
    return r.content


def main() -> int:
    args = parse_args()
    if not CSV_PATH.exists():
        print(f"ERROR: {CSV_PATH} not found. Download VGGSound CSV first.", file=sys.stderr)
        return 2

    # Load and index by class
    by_class: dict[str, list[tuple[str, str]]] = {c: [] for c in TARGET_CLASSES}
    with CSV_PATH.open() as f:
        for row in csv.reader(f):
            if len(row) < 4:
                continue
            yt_id, start, cls, split = row[0], row[1], row[2], row[3]
            if cls in by_class:
                by_class[cls].append((yt_id, start))

    print("VGGSound class candidate counts:")
    for cls, items in by_class.items():
        print(f"  {len(items):>5}  {cls}")
    print()

    rng = random.Random(args.seed)
    target = args.per_class
    oversample = args.oversample
    session = requests.Session()

    manifest: list[dict] = []
    fail_log: list[dict] = []

    for vgg_cls, coco_cls in TARGET_CLASSES.items():
        out_dir = OUT_ROOT / coco_cls
        out_dir.mkdir(parents=True, exist_ok=True)

        candidates = by_class[vgg_cls][:]
        rng.shuffle(candidates)
        pool = candidates[: target * oversample]

        kept = 0
        attempted = 0
        t0 = time.perf_counter()
        for yt_id, start in pool:
            if kept >= target:
                break
            attempted += 1
            content = fetch_one(yt_id, args.quality, session)
            if content is None:
                fail_log.append({"yt_id": yt_id, "vgg_class": vgg_cls, "reason": "placeholder_or_404"})
                continue
            out = out_dir / f"{yt_id}.jpg"
            out.write_bytes(content)
            manifest.append({
                "yt_id": yt_id,
                "start": start,
                "vgg_class": vgg_cls,
                "coco_class": coco_cls,
                "thumbnail": str(out.relative_to(Path("."))),
            })
            kept += 1

        elapsed = time.perf_counter() - t0
        print(f"  {coco_cls:<11} kept {kept}/{target}   attempted {attempted}   "
              f"link-rot {attempted - kept}   ({elapsed:.1f}s)")

    MANIFEST_PATH.write_text(json.dumps({
        "samples": manifest,
        "failed": fail_log,
        "target_per_class": target,
        "quality": args.quality,
        "seed": args.seed,
    }, indent=2))

    total_kept = len(manifest)
    print()
    print(f"Saved {total_kept} thumbnails. Manifest: {MANIFEST_PATH}")
    print(f"Link-rot rate (overall): {len(fail_log)/(len(fail_log)+total_kept)*100:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
