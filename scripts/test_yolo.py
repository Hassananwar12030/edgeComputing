"""
YOLO Labeler Smoke Test
=======================

Runs the existing `VisionProcessor` over data/test_images/ to confirm the
auto-labeling pipeline works end-to-end on the laptop before Pi deployment.

Usage:
    .venv/bin/python scripts/test_yolo.py
    .venv/bin/python scripts/test_yolo.py --image-dir other/path
    .venv/bin/python scripts/test_yolo.py --strategy largest_area

Exit code is non-zero if any image with an `expected_label` returns something else.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Make src/ importable when run from project root
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.edge.processing.vision import VisionProcessor  # noqa: E402


# Expected dominant label per fixture image when run with the project's
# default `classes_of_interest` filter (person, bicycle, car, motorcycle,
# bus, truck, dog, cat, bird). Used as a pass/fail check.
EXPECTED = {
    "bus.jpg":    "bus",     # bus + 3 persons, bus is most confident
    "zidane.jpg": "person",  # 2 persons
    "dog.jpg":    "dog",     # single dog (PyTorch hub sample)
    "eagle.jpg":  "bird",    # eagle → COCO 'bird' class
    "horses.jpg": None,      # YOLO detects 'horse', not in filter → filtered to None
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="models/yolov8n.pt",
                   help="Path to YOLOv8 weights")
    p.add_argument("--image-dir", default="data/test_images",
                   help="Directory of .jpg/.png images to score")
    p.add_argument("--confidence", type=float, default=0.5,
                   help="Confidence threshold")
    p.add_argument("--strategy", default="highest_confidence",
                   choices=["highest_confidence", "largest_area"],
                   help="Dominant-label selection strategy")
    p.add_argument("--no-class-filter", action="store_true",
                   help="Accept all 80 COCO classes (default: only classes_of_interest)")
    return p.parse_args()


def load_image(path: Path) -> np.ndarray:
    """Load image as RGB (ultralytics expects RGB or auto-handles BGR; we use RGB)."""
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise RuntimeError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def main() -> int:
    args = parse_args()

    image_dir = Path(args.image_dir)
    if not image_dir.is_dir():
        print(f"ERROR: image dir not found: {image_dir}", file=sys.stderr)
        return 2

    images = sorted([
        p for p in image_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ])
    if not images:
        print(f"ERROR: no images in {image_dir}", file=sys.stderr)
        return 2

    # `classes_of_interest=None` when --no-class-filter, else use the project default
    # which matches configs/edge_config.yaml's vision.classes_of_interest.
    classes = None if args.no_class_filter else [
        "person", "bicycle", "car", "motorcycle", "bus", "truck",
        "dog", "cat", "bird",
    ]

    print(f"Loading model: {args.model}")
    t0 = time.perf_counter()
    proc = VisionProcessor(
        model_path=args.model,
        confidence_threshold=args.confidence,
        classes_of_interest=classes,
    )
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"Model loaded in {load_ms:.0f} ms — {len(proc.get_class_names())} classes")
    print(f"Confidence threshold: {args.confidence}")
    print(f"Dominant-label strategy: {args.strategy}")
    print(f"Class filter: {classes if classes else 'ALL'}")
    print()

    # Warm up — first inference includes graph construction / kernel JIT on Apple Silicon.
    print("Warming up (1 inference)...")
    proc.detect_and_label(load_image(images[0]))

    results = []
    mismatches = 0
    print(f"{'IMAGE':<16} {'DOMINANT':<14} {'CONF':>6}  {'MS':>6}  DETECTIONS")
    print("-" * 90)
    for img_path in images:
        frame = load_image(img_path)
        t0 = time.perf_counter()
        # Re-implement detect_and_label inline to allow strategy override
        detections = proc.detect(frame)
        dom = proc.get_dominant_label(detections, strategy=args.strategy)
        ms = (time.perf_counter() - t0) * 1000

        label, conf = (dom if dom else (None, 0.0))
        det_summary = ", ".join(
            f"{d['class']}({d['confidence']:.2f})" for d in detections[:5]
        ) or "(none)"
        if len(detections) > 5:
            det_summary += f", +{len(detections) - 5} more"

        # Distinguish "no expectation" from "expected to be filtered to None".
        # Use sentinel via `in` check.
        mark = ""
        if img_path.name in EXPECTED:
            expected = EXPECTED[img_path.name]
            if label == expected:
                mark = " OK"
            else:
                mark = f" FAIL (expected {expected})"
                mismatches += 1

        print(f"{img_path.name:<16} {str(label):<14} {conf:>6.2f}  {ms:>6.1f}  {det_summary}{mark}")
        results.append((img_path.name, label, conf, ms, len(detections)))

    # Summary stats
    inference_ms = [r[3] for r in results]
    print()
    print("Summary")
    print("-------")
    print(f"  Images:              {len(results)}")
    print(f"  Mean inference:      {np.mean(inference_ms):.1f} ms")
    print(f"  Median inference:    {np.median(inference_ms):.1f} ms")
    print(f"  Min / Max:           {min(inference_ms):.1f} / {max(inference_ms):.1f} ms")
    print(f"  Mismatches vs expected: {mismatches}")

    return 0 if mismatches == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
