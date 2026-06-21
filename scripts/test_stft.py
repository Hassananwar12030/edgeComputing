"""
End-to-end STFT pipeline test on UrbanSound8K
==============================================

Verifies that:
1. WAV files load correctly from UrbanSound8K
2. STFTProcessor produces the expected (49, 128) mel-spectrogram for 500ms@16kHz
3. Spectrograms have plausible value ranges and visually distinguishable patterns

Strategy per file:
    load → resample to 16kHz → take center 500ms (8000 samples) → STFT → mel-spectrogram

Output:
    Stdout: per-class shape + statistics
    /tmp/stft_grid.png: 10-tile spectrogram grid, one sample per class
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.edge.processing.stft import STFTProcessor  # noqa: E402

DATASET = Path("data/urbansound8k/UrbanSound8K")
METADATA = DATASET / "metadata" / "UrbanSound8K.csv"
AUDIO_ROOT = DATASET / "audio"

# Match project config (configs/edge_config.yaml)
SAMPLE_RATE = 16000
CHUNK_SAMPLES = 8000  # 500 ms
EXPECTED_SHAPE = (51, 128)  # librosa center=True convention; see stft.py docstring


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="/tmp/stft_grid.png",
                   help="Where to save the 10-class spectrogram grid")
    return p.parse_args()


def pick_one_per_class(rows: list[dict]) -> dict[str, dict]:
    """Pick one sample per class — deterministic by taking the first occurrence."""
    by_class: dict[str, dict] = {}
    for r in rows:
        cls = r["class"]
        if cls not in by_class:
            by_class[cls] = r
    return by_class


def take_center_chunk(audio: np.ndarray, n_target: int) -> np.ndarray:
    """Return a fixed-length chunk centered on the audio — pad with zeros if shorter."""
    n = len(audio)
    if n >= n_target:
        start = (n - n_target) // 2
        return audio[start:start + n_target]
    # Pad symmetrically
    pad = n_target - n
    left = pad // 2
    right = pad - left
    return np.pad(audio, (left, right), mode="constant")


def main() -> int:
    args = parse_args()

    if not METADATA.exists():
        print(f"ERROR: {METADATA} not found", file=sys.stderr)
        return 2

    with METADATA.open() as f:
        rows = list(csv.DictReader(f))

    picks = pick_one_per_class(rows)
    proc = STFTProcessor(
        sample_rate=SAMPLE_RATE,
        n_fft=512,
        hop_length=160,
        n_mels=128,
        window_length=400,
        normalize=True,
    )

    print(f"{'CLASS':<22} {'NATIVE Hz':>9} {'DUR s':>7} {'SHAPE':>11} {'MIN':>6} {'MAX':>6} {'MEAN':>6}")
    print("-" * 80)

    grid_specs: list[tuple[str, np.ndarray]] = []
    failures = 0

    for cls in sorted(picks.keys()):
        r = picks[cls]
        wav_path = AUDIO_ROOT / f"fold{r['fold']}" / r["slice_file_name"]
        try:
            # librosa.load resamples to target sample rate automatically
            audio_native, native_sr = librosa.load(str(wav_path), sr=None, mono=True)
            audio_16k = librosa.resample(audio_native, orig_sr=native_sr, target_sr=SAMPLE_RATE)
            chunk = take_center_chunk(audio_16k, CHUNK_SAMPLES)
            spec = proc.process(chunk)
        except Exception as e:
            print(f"{cls:<22} FAIL: {e}")
            failures += 1
            continue

        shape_str = f"{spec.shape[0]}x{spec.shape[1]}"
        print(f"{cls:<22} {native_sr:>9} {len(audio_native)/native_sr:>7.2f} "
              f"{shape_str:>11} {spec.min():>6.2f} {spec.max():>6.2f} {spec.mean():>6.2f}")

        if spec.shape != EXPECTED_SHAPE:
            print(f"  WARN: shape {spec.shape} != expected {EXPECTED_SHAPE}")
            failures += 1

        grid_specs.append((cls, spec))

    # Visualization: 2x5 grid of spectrograms
    if grid_specs:
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        for ax, (cls, spec) in zip(axes.flat, grid_specs):
            # spec is (time, freq) — transpose to (freq, time) for display so
            # time goes horizontally and high freq is on top
            ax.imshow(spec.T, aspect="auto", origin="lower", cmap="magma")
            ax.set_title(cls, fontsize=10)
            ax.set_xlabel("frame (10ms)")
            ax.set_ylabel("mel bin")
        for ax in axes.flat[len(grid_specs):]:
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(args.output, dpi=100, bbox_inches="tight")
        print(f"\nSaved spectrogram grid: {args.output}")

    print()
    print(f"Result: {len(grid_specs)} processed, {failures} failures.")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
