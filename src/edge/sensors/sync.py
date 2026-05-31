"""
Sensor Synchronization Module
=============================

Synchronizes camera frames and audio chunks by timestamp.

Why synchronization matters:
- We use YOLOv8 on camera frames to generate labels for audio
- The label must match what was happening WHEN the audio was recorded
- Example: If a car passes by, we need the audio chunk from that moment

Synchronization approach:
1. Both sensors capture with timestamps
2. For each audio chunk, find the closest camera frame
3. Tolerance: ±50ms (audio perception threshold)

Timeline example:
    Audio:  |---chunk1---|---chunk2---|---chunk3---|
    Video:  |f1|  |f2|  |f3|  |f4|  |f5|  |f6|

    chunk2 at t=1.0s, f3 at t=0.98s, f4 at t=1.05s
    → f3 is closest (0.02s vs 0.05s) → use f3's label for chunk2

Usage:
    sync = SensorSynchronizer(camera, microphone)
    sync.start()
    frame, audio, timestamp = sync.capture_synchronized()
    sync.stop()
"""

import logging
import threading
import time
from collections import deque
from typing import Tuple, Optional, NamedTuple
import numpy as np


class SynchronizedSample(NamedTuple):
    """Container for synchronized audio-visual sample."""
    frame: np.ndarray          # Camera frame (H, W, 3)
    audio: np.ndarray          # Audio chunk (samples,)
    timestamp: float           # Reference timestamp
    frame_timestamp: float     # Actual frame capture time
    audio_timestamp: float     # Actual audio capture time
    sync_offset: float         # Time difference between frame and audio


class SensorSynchronizer:
    """
    Synchronizes camera and microphone capture.

    Maintains separate buffers for video frames and audio chunks,
    then matches them by timestamp when requested.

    The synchronizer runs capture loops in background threads
    and provides synchronized samples on demand.

    Attributes:
        camera: CameraCapture instance
        microphone: MicrophoneCapture instance
        sync_tolerance: Maximum allowed time difference (seconds)
        buffer_size: Number of samples to keep in each buffer
    """

    def __init__(
        self,
        camera,  # CameraCapture
        microphone,  # MicrophoneCapture
        sync_tolerance: float = 0.05,  # 50ms
        buffer_size: int = 30
    ):
        """
        Initialize the synchronizer.

        Args:
            camera: CameraCapture instance
            microphone: MicrophoneCapture instance
            sync_tolerance: Max time diff for valid sync (seconds)
            buffer_size: Size of frame/audio buffers
        """
        self.camera = camera
        self.microphone = microphone
        self.sync_tolerance = sync_tolerance
        self.buffer_size = buffer_size

        # Circular buffers for frames and audio
        self._frame_buffer = deque(maxlen=buffer_size)
        self._audio_buffer = deque(maxlen=buffer_size)

        # Threading
        self._capture_thread = None
        self._stop_event = threading.Event()

        self.is_running = False
        self.logger = logging.getLogger(__name__)

        # Statistics
        self.sync_stats = {
            'total_synced': 0,
            'sync_failures': 0,
            'avg_offset': 0.0
        }

    def start(self) -> None:
        """
        Start synchronized capture.

        Starts both sensors and begins background capture.
        """
        # Start individual sensors
        self.camera.start()
        self.microphone.start()

        # Start capture thread
        self._stop_event.clear()
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            daemon=True
        )
        self._capture_thread.start()

        self.is_running = True
        self.logger.info("Synchronized capture started")

    def stop(self) -> None:
        """
        Stop synchronized capture.
        """
        self._stop_event.set()

        if self._capture_thread:
            self._capture_thread.join(timeout=2.0)

        self.camera.stop()
        self.microphone.stop()

        self.is_running = False
        self.logger.info("Synchronized capture stopped")

    def _capture_loop(self) -> None:
        """
        Background capture loop.

        Continuously captures from both sensors and
        stores timestamped samples in buffers.
        """
        while not self._stop_event.is_set():
            try:
                # Capture frame
                frame, frame_ts = self.camera.capture()
                self._frame_buffer.append((frame, frame_ts))

                # Capture audio
                audio, audio_ts = self.microphone.capture(timeout=1.0)
                self._audio_buffer.append((audio, audio_ts))

            except Exception as e:
                self.logger.error(f"Capture error: {e}")
                time.sleep(0.1)

    def capture_synchronized(self) -> Optional[SynchronizedSample]:
        """
        Get a synchronized audio-visual sample.

        Matches the most recent audio chunk with the closest frame.

        Returns:
            SynchronizedSample if successful, None if sync failed

        Note:
            Returns None if:
            - Buffers are empty
            - No frame within sync_tolerance of audio
        """
        if not self._audio_buffer or not self._frame_buffer:
            return None

        # Get most recent audio chunk
        audio, audio_ts = self._audio_buffer[-1]

        # Find closest frame
        best_frame = None
        best_frame_ts = None
        min_offset = float('inf')

        for frame, frame_ts in self._frame_buffer:
            offset = abs(frame_ts - audio_ts)
            if offset < min_offset:
                min_offset = offset
                best_frame = frame
                best_frame_ts = frame_ts

        # Check if within tolerance
        if min_offset > self.sync_tolerance:
            self.sync_stats['sync_failures'] += 1
            self.logger.warning(
                f"Sync failed: offset {min_offset:.3f}s > "
                f"tolerance {self.sync_tolerance:.3f}s"
            )
            return None

        # Update statistics
        self.sync_stats['total_synced'] += 1
        n = self.sync_stats['total_synced']
        self.sync_stats['avg_offset'] = (
            (self.sync_stats['avg_offset'] * (n - 1) + min_offset) / n
        )

        return SynchronizedSample(
            frame=best_frame,
            audio=audio,
            timestamp=audio_ts,
            frame_timestamp=best_frame_ts,
            audio_timestamp=audio_ts,
            sync_offset=min_offset
        )

    def get_stats(self) -> dict:
        """Get synchronization statistics."""
        return self.sync_stats.copy()

    def clear_buffers(self) -> None:
        """Clear both capture buffers."""
        self._frame_buffer.clear()
        self._audio_buffer.clear()

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False
