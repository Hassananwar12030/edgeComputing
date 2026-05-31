"""
Microphone Capture Module
=========================

Handles I2S MEMS microphone capture on Raspberry Pi.

I2S (Inter-IC Sound) is a protocol for digital audio:
- Data line (DOUT): Carries the actual audio samples
- Clock line (BCLK): Bit clock for synchronization
- Word select (LRCLK): Indicates left/right channel

Common I2S MEMS microphones for Raspberry Pi:
- INMP441 (most common, good quality)
- SPH0645 (Adafruit)
- ICS-43432

Audio specifications for this project:
- Sample rate: 16,000 Hz (16 kHz) - standard for speech/audio ML
- Bit depth: 16-bit signed integers
- Channels: 1 (mono)
- Chunk duration: 500ms (8000 samples at 16kHz)

Usage:
    mic = MicrophoneCapture(sample_rate=16000, chunk_duration=0.5)
    mic.start()
    audio_chunk, timestamp = mic.capture()  # Returns (8000,) numpy array
    mic.stop()

Note:
    I2S requires kernel configuration on Raspberry Pi.
    See docs/HARDWARE_SETUP.md for configuration steps.
"""

import logging
import time
import queue
import threading
from typing import Tuple, Optional
import numpy as np

# sounddevice is used for audio capture
# It wraps PortAudio and works across platforms
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    sd = None


class MicrophoneCapture:
    """
    I2S microphone capture wrapper.

    Provides continuous audio capture with buffering.
    Audio is captured in a background thread and chunks
    are retrieved via the capture() method.

    Attributes:
        sample_rate: Audio sample rate in Hz
        chunk_duration: Duration of each chunk in seconds
        chunk_samples: Number of samples per chunk
        dtype: numpy dtype for audio samples
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_duration: float = 0.5,
        device: Optional[int] = None,
        channels: int = 1,
        dtype: str = "int16"
    ):
        """
        Initialize microphone capture.

        Args:
            sample_rate: Audio sample rate (default 16kHz)
            chunk_duration: Duration of each chunk in seconds
            device: Audio device index (None for default)
            channels: Number of channels (1 for mono)
            dtype: Data type for samples
        """
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_samples = int(sample_rate * chunk_duration)
        self.device = device
        self.channels = channels
        self.dtype = dtype

        # Audio buffer and state
        self._audio_queue = queue.Queue()
        self._stream = None
        self.is_running = False
        self.logger = logging.getLogger(__name__)

        # For development/testing without microphone
        self._mock_mode = not SOUNDDEVICE_AVAILABLE

        if self._mock_mode:
            self.logger.warning(
                "sounddevice not available - running in mock mode. "
                "Install with: pip install sounddevice"
            )

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info,
        status
    ) -> None:
        """
        Callback function called by sounddevice for each audio block.

        This runs in a separate thread and puts audio chunks into a queue.

        Args:
            indata: Input audio data
            frames: Number of frames
            time_info: Timing information
            status: Status flags
        """
        if status:
            self.logger.warning(f"Audio callback status: {status}")

        # Put a copy of the data into the queue
        # (indata is reused by sounddevice)
        self._audio_queue.put((indata.copy(), time.time()))

    def start(self) -> None:
        """
        Start audio capture.

        Begins background capture thread that fills the audio queue.
        """
        if self._mock_mode:
            self.is_running = True
            self.logger.info("Mock microphone started")
            return

        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=self.chunk_samples,
                device=self.device,
                channels=self.channels,
                dtype=self.dtype,
                callback=self._audio_callback
            )
            self._stream.start()
            self.is_running = True
            self.logger.info(
                f"Microphone started: {self.sample_rate}Hz, "
                f"{self.chunk_duration}s chunks ({self.chunk_samples} samples)"
            )

        except Exception as e:
            self.logger.error(f"Failed to start microphone: {e}")
            raise

    def stop(self) -> None:
        """
        Stop audio capture and release resources.
        """
        if self._mock_mode:
            self.is_running = False
            self.logger.info("Mock microphone stopped")
            return

        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            self.is_running = False
            self.logger.info("Microphone stopped")

    def capture(self, timeout: float = 1.0) -> Tuple[np.ndarray, float]:
        """
        Get the next audio chunk.

        Blocks until audio is available or timeout.

        Args:
            timeout: Maximum time to wait for audio

        Returns:
            Tuple of (audio_chunk, timestamp) where:
            - audio_chunk: numpy array of shape (chunk_samples,), dtype int16
            - timestamp: time.time() when chunk was captured

        Raises:
            RuntimeError: If microphone not started
            queue.Empty: If no audio available within timeout
        """
        if not self.is_running:
            raise RuntimeError("Microphone not started. Call start() first.")

        if self._mock_mode:
            # Generate mock audio for development
            audio = self._generate_mock_audio()
            return audio, time.time()

        # Get audio from queue (blocks until available)
        audio, timestamp = self._audio_queue.get(timeout=timeout)

        # Flatten to 1D if needed (mono)
        if audio.ndim > 1:
            audio = audio.flatten()

        return audio, timestamp

    def _generate_mock_audio(self) -> np.ndarray:
        """
        Generate mock audio for development/testing.

        Creates a simple sine wave with some noise,
        useful for testing the STFT pipeline.
        """
        t = np.linspace(0, self.chunk_duration, self.chunk_samples)

        # Generate sine wave at 440 Hz (A4 note)
        frequency = 440
        amplitude = 10000  # For int16

        audio = amplitude * np.sin(2 * np.pi * frequency * t)

        # Add some noise
        noise = np.random.normal(0, amplitude * 0.1, len(audio))
        audio = audio + noise

        # Convert to int16
        audio = audio.astype(np.int16)

        # Simulate processing time
        time.sleep(self.chunk_duration * 0.9)

        return audio

    def clear_buffer(self) -> None:
        """Clear any buffered audio chunks."""
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

    def get_sample_rate(self) -> int:
        """Get the current sample rate."""
        return self.sample_rate

    def get_chunk_duration(self) -> float:
        """Get the chunk duration in seconds."""
        return self.chunk_duration

    @staticmethod
    def list_devices() -> None:
        """Print available audio devices."""
        if SOUNDDEVICE_AVAILABLE:
            print(sd.query_devices())
        else:
            print("sounddevice not available")

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False
