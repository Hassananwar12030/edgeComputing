"""
Measure weak-supervision label noise
====================================

Validates the thesis assumption: when audio is labeled X by YOLO-on-camera-frame,
does YOLO actually detect X in the frame? Run YOLO on a set of VGGSound thumbnails
(human-labeled by audio content) and compute agreement vs. those labels.

Inputs:
    data/vggsound/manifest.json  (from fetch_vggsound_thumbnails.py)
    models/yolov8n.pt

Outputs:
    data/vggsound/label_noise_results.json  (per-sample + summary)
    stdout: per-class agreement, confusion matrix, abstention rate

Usage:
    .venv/bin/python scripts/measure_label_noise.py
    .venv/bin/python scripts/measure_label_noise.py --confidence 0.3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.edge.processing.vision import VisionProcessor  # noqa: E402


MANIFEST_PATH = Path("data/vggsound/manifest.json")
RESULTS_PATH = Path("data/vggsound/label_noise_results.json")

# These are the COCO classes we care about for the thesis (subset of the 80).
# When YOLO detects something outside this set, we count it as "out_of_scope"
# rather than a wrong label — same behavior the real edge pipeline will have.
COCO_TARGET_CLASSES = {"car", "motorcycle", "dog", "cat", "bird"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="models/yolov8n.pt")
    p.add_argument("--confidence", type=float, default=0.5,
                   help="YOLO confidence threshold")
    p.add_argument("--strategy", default="highest_confidence",
                   choices=["highest_confidence", "largest_area"])
    return p.parse_args()


def load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise RuntimeError(f"Cannot read {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def main() -> int:
    args = parse_args()

    if not MANIFEST_PATH.exists():
        print(f"ERROR: {MANIFEST_PATH} not found. Run fetch_vggsound_thumbnails.py first.",
              file=sys.stderr)
        return 2
    manifest = json.loads(MANIFEST_PATH.read_text())
    samples = manifest["samples"]
    if not samples:
        print("ERROR: manifest has no samples", file=sys.stderr)
        return 2

    # Use a class filter so the measurement matches what the real edge pipeline
    # will see — anything outside is dropped to None.
    proc = VisionProcessor(
        model_path=args.model,
        confidence_threshold=args.confidence,
        classes_of_interest=sorted(COCO_TARGET_CLASSES),
    )
    print(f"Model: {args.model}  confidence={args.confidence}  strategy={args.strategy}")
    print(f"Samples: {len(samples)}\n")

    per_sample = []
    # confusion[true_label][predicted_label] = count, where predicted can be a
    # COCO class, "no_detection" (filter dropped everything / nothing detected),
    # or the literal None object.
    confusion: dict[str, Counter] = defaultdict(Counter)

    t_total = time.perf_counter()
    for i, s in enumerate(samples, 1):
        path = Path(s["thumbnail"])
        true_coco = s["coco_class"]

        try:
            frame = load_rgb(path)
        except Exception as e:
            print(f"  [{i:>3}/{len(samples)}] {path.name} READ FAIL: {e}")
            continue

        detections = proc.detect(frame)
        dom = proc.get_dominant_label(detections, strategy=args.strategy)
        predicted = dom[0] if dom else "no_detection"
        confusion[true_coco][predicted] += 1
        per_sample.append({
            "yt_id": s["yt_id"],
            "true_coco": true_coco,
            "vgg_class": s["vgg_class"],
            "predicted": predicted,
            "confidence": dom[1] if dom else 0.0,
            "n_detections": len(detections),
        })

        if i % 25 == 0:
            print(f"  scored {i}/{len(samples)}...")

    elapsed = time.perf_counter() - t_total
    print(f"\nScored {len(per_sample)} samples in {elapsed:.1f}s "
          f"({elapsed*1000/max(len(per_sample),1):.0f} ms/sample)\n")

    # ---------- Summary ----------
    print("PER-CLASS AGREEMENT")
    print("-" * 80)
    print(f"{'TRUE':<12} {'N':>5}  {'CORRECT':>8}  {'AGREE %':>8}  {'NO-DET %':>9}  WRONG (counts)")
    print("-" * 80)

    overall_correct = 0
    overall_no_det = 0
    overall_n = 0
    for true_cls in sorted(COCO_TARGET_CLASSES):
        row = confusion[true_cls]
        n = sum(row.values())
        if n == 0:
            continue
        correct = row[true_cls]
        no_det = row["no_detection"]
        wrong = {k: v for k, v in row.items() if k not in (true_cls, "no_detection")}
        wrong_str = ", ".join(f"{k}={v}" for k, v in sorted(wrong.items(), key=lambda x: -x[1])) or "-"
        agree_pct = correct / n * 100
        no_det_pct = no_det / n * 100
        print(f"{true_cls:<12} {n:>5}  {correct:>8}  {agree_pct:>7.1f}%  {no_det_pct:>8.1f}%  {wrong_str}")
        overall_correct += correct
        overall_no_det += no_det
        overall_n += n

    print("-" * 80)
    if overall_n:
        print(f"{'OVERALL':<12} {overall_n:>5}  {overall_correct:>8}  "
              f"{overall_correct/overall_n*100:>7.1f}%  "
              f"{overall_no_det/overall_n*100:>8.1f}%")

    print()
    print("INTERPRETATION GUIDE")
    print("-" * 80)
    print("  AGREE %  = thumbnail YOLO label matches VGGSound audio label  (true positive)")
    print("  NO-DET % = YOLO found nothing in target classes (silent in pipeline = skipped)")
    print("  WRONG    = YOLO confidently labeled the frame as something else (label noise)")
    print()
    print("In the real pipeline NO-DETECTION samples are dropped (no training pair created),")
    print("so they aren't 'noise' in the harmful sense — they're missed yield. WRONG cells")
    print("ARE harmful: a training pair will be created with an incorrect label.")

    RESULTS_PATH.write_text(json.dumps({
        "config": {
            "model": args.model,
            "confidence": args.confidence,
            "strategy": args.strategy,
        },
        "per_sample": per_sample,
        "confusion": {k: dict(v) for k, v in confusion.items()},
        "summary": {
            "total": overall_n,
            "agree": overall_correct,
            "no_detection": overall_no_det,
            "agree_pct": overall_correct / overall_n * 100 if overall_n else 0.0,
            "no_detection_pct": overall_no_det / overall_n * 100 if overall_n else 0.0,
        },
    }, indent=2))
    print(f"\nFull results: {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
