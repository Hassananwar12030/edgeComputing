"""
FusionModel build and forward-pass smoke test
==============================================

Verifies that:
1. FusionModel builds with the corrected audio shape (51, 128, 1)
2. MobileNetV3-Small (vision branch) downloads weights and integrates
3. Two-input forward pass produces (batch, num_classes) output

Not used in step 3 bootstrap (which is audio-only on UrbanSound8K) — this
exists to close the verification gap created when scripts/test_stft.py's
bulk shape update touched fusion_model.py.

First run downloads MobileNetV3-Small ImageNet weights (~10 MB).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# Fix SSL chain for Keras' weight downloader on corporate networks
import truststore
truststore.inject_into_ssl()

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.server.training.models.fusion_model import FusionModel  # noqa: E402

AUDIO_SHAPE = (51, 128, 1)
VISION_SHAPE = (224, 224, 3)
NUM_CLASSES = 10
BATCH = 2


def main() -> int:
    print("Building FusionModel — first run pulls MobileNetV3-Small ImageNet weights...")
    model = FusionModel(num_classes=NUM_CLASSES)
    n_total = model.count_params()
    n_train = sum(np.prod(w.shape) for w in model.trainable_weights)

    print(f"\nTotal params:        {n_total:>10,}  ({n_total/1e6:.2f} M)")
    print(f"Trainable params:    {int(n_train):>10,}  ({n_train/1e6:.2f} M)")
    print(f"Non-trainable (frozen MobileNet): {n_total - int(n_train):>10,}")
    print(f"Input shapes:        {[inp.shape for inp in model.inputs]}")
    print(f"Output shape:        {model.output_shape}")

    # Forward pass
    rand_vis = np.random.rand(BATCH, *VISION_SHAPE).astype(np.float32)
    rand_aud = np.random.rand(BATCH, *AUDIO_SHAPE).astype(np.float32)
    out = model.predict([rand_vis, rand_aud], verbose=0)

    print(f"\nForward pass on batch={BATCH}:")
    print(f"  vision input shape: {rand_vis.shape}")
    print(f"  audio input shape:  {rand_aud.shape}")
    print(f"  output shape:       {out.shape}")
    print(f"  row sums:           {out.sum(axis=1)}  (softmax → ~1.0)")

    ok = out.shape == (BATCH, NUM_CLASSES) and np.allclose(out.sum(axis=1), 1.0, atol=1e-5)
    print(f"\nResult: {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
