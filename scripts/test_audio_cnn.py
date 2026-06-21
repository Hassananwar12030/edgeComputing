"""
AudioCNN model build and forward-pass smoke test
=================================================

Verifies that:
1. AudioCNN (classifier) builds with the corrected (51, 128, 1) input shape
2. AudioCNNFeatureExtractor (fusion-branch variant) builds
3. A forward pass on a *real* mel-spectrogram from UrbanSound8K produces a
   valid (1, num_classes) probability distribution

Output:
    stdout: model summaries, parameter counts, output shapes, sanity checks

This is a structural test, not a trained model — predictions will be random
(softmax over random weights) until step 4 (training).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

# Quiet TF info logs
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import librosa
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.edge.processing.stft import STFTProcessor  # noqa: E402
from src.server.training.models.audio_cnn import (  # noqa: E402
    AudioCNN, AudioCNNFeatureExtractor,
)

EXPECTED_INPUT = (51, 128, 1)
NUM_CLASSES = 10
SAMPLE_RATE = 16000
CHUNK_SAMPLES = 8000

DATASET = Path("data/urbansound8k/UrbanSound8K")
METADATA = DATASET / "metadata" / "UrbanSound8K.csv"
AUDIO_ROOT = DATASET / "audio"


def real_spectrogram() -> np.ndarray:
    """Load one UrbanSound8K clip → 16kHz → center 500ms → STFT."""
    with METADATA.open() as f:
        rows = list(csv.DictReader(f))
    r = rows[0]  # deterministic: first row of the metadata
    wav = AUDIO_ROOT / f"fold{r['fold']}" / r["slice_file_name"]
    audio, native_sr = librosa.load(str(wav), sr=None, mono=True)
    audio = librosa.resample(audio, orig_sr=native_sr, target_sr=SAMPLE_RATE)
    n = len(audio)
    if n >= CHUNK_SAMPLES:
        start = (n - CHUNK_SAMPLES) // 2
        chunk = audio[start:start + CHUNK_SAMPLES]
    else:
        pad = CHUNK_SAMPLES - n
        chunk = np.pad(audio, (pad // 2, pad - pad // 2))
    proc = STFTProcessor()
    return proc.process(chunk), r["class"]


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def check_classifier() -> bool:
    section("AudioCNN — classifier (used for bootstrap on UrbanSound8K)")

    model = AudioCNN(input_shape=EXPECTED_INPUT, num_classes=NUM_CLASSES)
    model.summary(line_length=80)

    n_params = model.count_params()
    print(f"\n  Trainable params: {n_params:,}")
    print(f"  Input shape : {model.input_shape}")
    print(f"  Output shape: {model.output_shape}")

    # Forward pass with random data
    rand = np.random.rand(2, *EXPECTED_INPUT).astype(np.float32)
    out = model.predict(rand, verbose=0)
    print(f"\n  Forward pass on random batch shape={rand.shape}:")
    print(f"    output shape:    {out.shape}")
    print(f"    row sums:        {out.sum(axis=1)}  (should be ~1.0 each — softmax)")
    print(f"    output range:    [{out.min():.4f}, {out.max():.4f}]")

    if out.shape != (2, NUM_CLASSES):
        print(f"  FAIL: expected output shape (2, {NUM_CLASSES})")
        return False
    if not np.allclose(out.sum(axis=1), 1.0, atol=1e-5):
        print(f"  FAIL: softmax rows don't sum to 1")
        return False

    # Forward pass with a real spectrogram
    spec, cls = real_spectrogram()
    batch = spec.reshape(1, *EXPECTED_INPUT).astype(np.float32)
    out = model.predict(batch, verbose=0)
    pred_idx = int(np.argmax(out[0]))
    print(f"\n  Forward pass on REAL UrbanSound8K sample (class={cls!r}):")
    print(f"    output shape: {out.shape}")
    print(f"    argmax index: {pred_idx}  (random — model is untrained)")
    print(f"    confidence:   {out[0, pred_idx]:.3f}")
    return True


def check_feature_extractor() -> bool:
    section("AudioCNNFeatureExtractor — backbone for fusion model")

    model = AudioCNNFeatureExtractor(input_shape=EXPECTED_INPUT, output_dim=128)
    model.summary(line_length=80)
    n_params = model.count_params()
    print(f"\n  Trainable params: {n_params:,}")
    print(f"  Input shape : {model.input_shape}")
    print(f"  Output shape: {model.output_shape}")

    rand = np.random.rand(4, *EXPECTED_INPUT).astype(np.float32)
    out = model.predict(rand, verbose=0)
    print(f"\n  Forward pass on shape={rand.shape}:")
    print(f"    output shape: {out.shape}  (expected (4, 128))")

    if out.shape != (4, 128):
        print(f"  FAIL: expected (4, 128)")
        return False
    return True


def main() -> int:
    ok1 = check_classifier()
    ok2 = check_feature_extractor()
    section("Result")
    print(f"  AudioCNN classifier:     {'OK' if ok1 else 'FAIL'}")
    print(f"  AudioCNN feature extr.:  {'OK' if ok2 else 'FAIL'}")
    return 0 if (ok1 and ok2) else 1


if __name__ == "__main__":
    sys.exit(main())
