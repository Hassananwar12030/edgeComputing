"""
Sensors Module
==============

This module handles all sensor capture on the Raspberry Pi:
- Camera capture using picamera2
- Microphone capture using I2S interface
- Synchronization of audio and video streams

The synchronization is critical because we need to match audio chunks
with the visual content at the same moment in time.

Classes:
    CameraCapture: Manages camera hardware and frame capture
    MicrophoneCapture: Manages I2S microphone and audio capture
    SensorSynchronizer: Aligns audio and video by timestamp
"""

from .camera import CameraCapture
from .microphone import MicrophoneCapture
from .sync import SensorSynchronizer

__all__ = ["CameraCapture", "MicrophoneCapture", "SensorSynchronizer"]
