"""
Model Architectures
===================

Neural network architectures for the fusion model.

Models:
    AudioCNN: CNN for mel-spectrogram classification
    VisionCNN: MobileNetV3-based vision model
    FusionModel: Combined audio-visual model

The fusion model architecture:

    Vision Input (224×224×3)      Audio Input (51×128×1)
           │                              │
           ▼                              ▼
    ┌─────────────────┐           ┌─────────────────┐
    │  MobileNetV3    │           │   Audio CNN     │
    │  (pre-trained)  │           │  (from scratch) │
    └────────┬────────┘           └────────┬────────┘
             │                             │
         256-dim                       128-dim
             │                             │
             └──────────┬──────────────────┘
                        ▼
                 ┌─────────────────┐
                 │  Concatenate    │
                 │    (384-dim)    │
                 └────────┬────────┘
                          ▼
                 ┌─────────────────┐
                 │  Fusion Head    │
                 └────────┬────────┘
                          ▼
                     Prediction
"""

from .audio_cnn import AudioCNN
from .fusion_model import FusionModel

__all__ = ["AudioCNN", "FusionModel"]
