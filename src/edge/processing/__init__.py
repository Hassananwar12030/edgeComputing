"""
Processing Module
=================

This module handles all signal processing on the edge device:
- STFT/Mel-spectrogram generation from audio
- YOLOv8 inference for auto-labeling
- Feature extraction for the fusion model

The processing pipeline:
    Audio chunk (16kHz, 500ms) → STFT → Mel-spectrogram (51, 128)
    Camera frame (640x480) → YOLOv8 → Labels + Bounding boxes

Classes:
    STFTProcessor: Converts audio to mel-spectrograms
    VisionProcessor: Runs YOLOv8 for object detection
    FeatureExtractor: Extracts features for fusion model
"""

from .stft import STFTProcessor
from .vision import VisionProcessor

__all__ = ["STFTProcessor", "VisionProcessor"]
