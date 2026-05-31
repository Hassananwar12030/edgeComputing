"""
Camera Capture Module
=====================

Handles camera initialization and frame capture on Raspberry Pi using picamera2.

The Camera Module 3 provides:
- 12MP sensor
- Autofocus capability
- HDR support
- Up to 120fps at lower resolutions

For this project, we use:
- Resolution: 640x480 (balance between detail and processing speed)
- Frame rate: 10 FPS (matched to audio capture rate)
- Format: RGB for YOLOv8 compatibility

Usage:
    camera = CameraCapture(width=640, height=480, fps=10)
    camera.start()
    frame = camera.capture()  # Returns numpy array (H, W, 3)
    camera.stop()

Note:
    picamera2 is the official library for Raspberry Pi cameras on Bookworm OS.
    It replaces the legacy picamera library.
"""

import logging
import time
from typing import Tuple, Optional
import numpy as np

# picamera2 is only available on Raspberry Pi
# We use conditional import for development on other platforms
try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False
    Picamera2 = None


class CameraCapture:
    """
    Camera capture wrapper for Raspberry Pi Camera Module 3.

    This class provides a simple interface to:
    1. Initialize the camera with specified settings
    2. Capture frames as numpy arrays
    3. Get timestamps for synchronization
    4. Handle camera lifecycle (start/stop)

    Attributes:
        width: Frame width in pixels
        height: Frame height in pixels
        fps: Target frames per second
        camera: picamera2 instance (None on non-Pi platforms)
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 10,
        format: str = "RGB888"
    ):
        """
        Initialize camera capture.

        Args:
            width: Frame width in pixels
            height: Frame height in pixels
            fps: Target capture rate
            format: Pixel format (RGB888 for RGB numpy arrays)
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.format = format

        self.camera = None
        self.is_running = False
        self.logger = logging.getLogger(__name__)

        # For development/testing without camera
        self._mock_mode = not PICAMERA_AVAILABLE

        if self._mock_mode:
            self.logger.warning(
                "picamera2 not available - running in mock mode. "
                "This is normal for development on non-Pi platforms."
            )

    def start(self) -> None:
        """
        Start the camera.

        Initializes the camera hardware and begins capture.
        Call this before calling capture().
        """
        if self._mock_mode:
            self.logger.info("Mock camera started")
            self.is_running = True
            return

        try:
            self.camera = Picamera2()

            # Configure camera
            # main: The primary stream configuration
            # lores: Low-resolution stream (not used here)
            config = self.camera.create_preview_configuration(
                main={
                    "size": (self.width, self.height),
                    "format": self.format
                }
            )
            self.camera.configure(config)

            # Set frame rate via frame duration (microseconds)
            # frame_duration = 1/fps * 1_000_000
            frame_duration_us = int(1_000_000 / self.fps)
            self.camera.set_controls({
                "FrameDurationLimits": (frame_duration_us, frame_duration_us)
            })

            self.camera.start()
            self.is_running = True
            self.logger.info(
                f"Camera started: {self.width}x{self.height} @ {self.fps}fps"
            )

        except Exception as e:
            self.logger.error(f"Failed to start camera: {e}")
            raise

    def stop(self) -> None:
        """
        Stop the camera and release resources.

        Always call this when done to properly release hardware.
        """
        if self._mock_mode:
            self.is_running = False
            self.logger.info("Mock camera stopped")
            return

        if self.camera:
            self.camera.stop()
            self.camera.close()
            self.camera = None
            self.is_running = False
            self.logger.info("Camera stopped")

    def capture(self) -> Tuple[np.ndarray, float]:
        """
        Capture a single frame.

        Returns:
            Tuple of (frame, timestamp) where:
            - frame: numpy array of shape (height, width, 3), dtype uint8
            - timestamp: time.time() when frame was captured

        Raises:
            RuntimeError: If camera not started
        """
        if not self.is_running:
            raise RuntimeError("Camera not started. Call start() first.")

        timestamp = time.time()

        if self._mock_mode:
            # Generate mock frame for development
            frame = self._generate_mock_frame()
            return frame, timestamp

        # Capture actual frame
        frame = self.camera.capture_array()

        return frame, timestamp

    def _generate_mock_frame(self) -> np.ndarray:
        """
        Generate a mock frame for development/testing.

        Creates a simple pattern that changes over time,
        useful for testing the pipeline without actual camera.
        """
        # Create gradient pattern
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Add some variation based on time
        t = int(time.time() * 10) % 256

        # Simple gradient
        for y in range(self.height):
            for x in range(self.width):
                frame[y, x, 0] = (x + t) % 256  # R
                frame[y, x, 1] = (y + t) % 256  # G
                frame[y, x, 2] = ((x + y) + t) % 256  # B

        return frame

    def get_resolution(self) -> Tuple[int, int]:
        """Get current resolution as (width, height)."""
        return (self.width, self.height)

    def get_fps(self) -> int:
        """Get current frame rate."""
        return self.fps

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False
