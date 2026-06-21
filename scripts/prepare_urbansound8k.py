"""
Preprocess UrbanSound8K into a cached numpy bundle.
====================================================

For each WAV:
    load → resample to 16kHz → center 500ms → STFT → (51,128) mel-spectrogram

Fold split (single-fold eval — full 10-fold CV is for the thesis report later):
    folds 1-8 → train
    fold 9    → validation
    fold 10   → test

Output:
    data/urbansound8k/cache.npz with X_train, y_train, X_val, y_val, X_test, y_test, classes

Re-running with --force overwrites the cache; otherwise skipped if cache exists.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import librosa
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.edge.processing.stft import STFTProcessor  # noqa: E402

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 8000  # 500 ms

DATASET = Path("data/urbansound8k/UrbanSound8K")
METADATA = DATASET / "metadata" / "UrbanSound8K.csv"
AUDIO_ROOT = DATASET / "audio"
CACHE = Path("data/urbansound8k/cache.npz")

TRAIN_FOLDS = set(range(1, 9))   # 1..8
VAL_FOLDS = {9}
TEST_FOLDS = {10}

# Sliding-window augmentation: how many 500ms windows to extract per *training*
# clip (val/test always stay center-crop = 1 window per clip, for fair compare).
TRAIN_WINDOWS_PER_CLIP = 4


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--force", action="store_true",
                   help="Regenerate cache even if it exists")
    p.add_argument("--out", default=str(CACHE),
                   help="Output cache .npz path (default: data/urbansound8k/cache.npz)")
    p.add_argument("--train-windows", type=int, default=TRAIN_WINDOWS_PER_CLIP,
                   help="Up to N evenly-spaced 500ms windows per training clip "
                        "(val/test always 1 center-crop window)")
    return p.parse_args()


def take_center_chunk(audio: np.ndarray, n_target: int) -> np.ndarray:
    """Single center 500ms slice — used for val/test and short training clips."""
    n = len(audio)
    if n >= n_target:
        start = (n - n_target) // 2
        return audio[start:start + n_target]
    pad = n_target - n
    return np.pad(audio, (pad // 2, pad - pad // 2))


def extract_train_windows(audio: np.ndarray, n_target: int, max_windows: int) -> list[np.ndarray]:
    """
    Up to `max_windows` 500ms windows from one training clip, evenly distributed.

    Rules:
      - If the clip is shorter than the window size, pad-center to one window.
      - Otherwise pick K windows where K = min(max_windows, floor(len / window)).
        Window starts are evenly spaced from 0 to (len - window_size).
      - Very-short clips therefore give 1 window; full 4s clips give max_windows.

    This is non-random and deterministic — caching this lets us train multiple
    epochs over the same N×bigger set without recomputing every epoch.
    """
    n = len(audio)
    if n < n_target:
        pad = n_target - n
        return [np.pad(audio, (pad // 2, pad - pad // 2))]

    n_fits = max(1, n // n_target)
    n_take = max(1, min(max_windows, n_fits))

    if n_take == 1:
        return [take_center_chunk(audio, n_target)]

    last_start = n - n_target
    starts = np.linspace(0, last_start, n_take).astype(int)
    return [audio[s:s + n_target] for s in starts]


def main() -> int:
    args = parse_args()

    cache = Path(args.out)
    if cache.exists() and not args.force:
        print(f"Cache exists at {cache} — use --force to regenerate.")
        # Print summary
        data = np.load(cache, allow_pickle=True)
        for split in ("train", "val", "test"):
            X = data[f"X_{split}"]
            y = data[f"y_{split}"]
            print(f"  {split:<6}  X={X.shape}  y={y.shape}")
        return 0

    if not METADATA.exists():
        print(f"ERROR: {METADATA} not found", file=sys.stderr)
        return 2

    with METADATA.open() as f:
        rows = list(csv.DictReader(f))

    classes = sorted({r["class"] for r in rows})
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    print(f"Classes ({len(classes)}): {classes}")
    print(f"Total samples: {len(rows)}\n")

    proc = STFTProcessor(
        sample_rate=SAMPLE_RATE,
        n_fft=512,
        hop_length=160,
        n_mels=128,
        window_length=400,
        normalize=True,
    )

    splits: dict[str, list[tuple[np.ndarray, int]]] = {"train": [], "val": [], "test": []}
    skipped = 0
    t0 = time.perf_counter()

    for i, r in enumerate(rows, 1):
        fold = int(r["fold"])
        wav = AUDIO_ROOT / f"fold{fold}" / r["slice_file_name"]
        try:
            audio, native_sr = librosa.load(str(wav), sr=None, mono=True)
            audio_16k = librosa.resample(audio, orig_sr=native_sr, target_sr=SAMPLE_RATE)
        except Exception as e:
            print(f"  SKIP {r['slice_file_name']}: {e}")
            skipped += 1
            continue

        label = cls_to_idx[r["class"]]
        if fold in TRAIN_FOLDS:
            # Sliding-window augmentation for training only
            chunks = extract_train_windows(audio_16k, CHUNK_SAMPLES, args.train_windows)
            for chunk in chunks:
                spec = proc.process(chunk).astype(np.float32)
                splits["train"].append((spec, label))
        elif fold in VAL_FOLDS:
            chunk = take_center_chunk(audio_16k, CHUNK_SAMPLES)
            spec = proc.process(chunk).astype(np.float32)
            splits["val"].append((spec, label))
        elif fold in TEST_FOLDS:
            chunk = take_center_chunk(audio_16k, CHUNK_SAMPLES)
            spec = proc.process(chunk).astype(np.float32)
            splits["test"].append((spec, label))

        if i % 500 == 0:
            elapsed = time.perf_counter() - t0
            rate = i / elapsed
            eta = (len(rows) - i) / max(rate, 1e-6)
            print(f"  {i:>5}/{len(rows)}   {rate:>5.0f} samples/s   ETA {eta:>5.0f}s")

    elapsed = time.perf_counter() - t0
    print(f"\nProcessed {len(rows) - skipped} samples in {elapsed:.1f}s "
          f"({(len(rows) - skipped) / elapsed:.0f}/s) — skipped {skipped}")

    # Stack and add channel dim → (N, 51, 128, 1)
    out: dict[str, np.ndarray] = {"classes": np.array(classes)}
    for name, items in splits.items():
        if not items:
            continue
        X = np.stack([x for x, _ in items])[..., np.newaxis]
        y = np.array([lbl for _, lbl in items], dtype=np.int32)
        out[f"X_{name}"] = X.astype(np.float32)
        out[f"y_{name}"] = y

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **out)
    cache_mb = cache.stat().st_size / 1e6
    print(f"\nSaved {cache} ({cache_mb:.1f} MB)")

    # Distribution summary
    print(f"\n{'SPLIT':<6} {'N':>6}  " + "  ".join(f"{c:<22}" for c in classes))
    for split in ("train", "val", "test"):
        if f"X_{split}" not in out:
            continue
        y = out[f"y_{split}"]
        n = len(y)
        counts = np.bincount(y, minlength=len(classes))
        row = " ".join(f"{cnt:>22}" for cnt in counts)
        print(f"{split:<6} {n:>6}   {row}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
