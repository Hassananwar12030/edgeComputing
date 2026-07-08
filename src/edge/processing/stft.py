"""
STFT and Mel-Spectrogram Processing
===================================

Converts raw audio waveforms to mel-spectrograms for the audio classifier.

Background:
- STFT (Short-Time Fourier Transform) analyzes frequency content over time
- Mel scale mimics human hearing (more resolution at lower frequencies)
- Spectrograms are 2D representations: time × frequency

Processing parameters (from research standards):
- Sample rate: 16,000 Hz
- Window size: 400 samples (25ms at 16kHz)
- Hop length: 160 samples (10ms) → 63% overlap
- FFT size: 512 (next power of 2 after window)
- Mel bins: 128 (standard for audio classification)

For 500ms audio (8000 samples) with librosa default (`center=True`):
- librosa pads so frame i is *centered* at sample i*hop_length, adding edge
  frames → 51 frames total (this is the audio-ML convention)
- Output shape: (51, 128) or (51, 128, 1) with channel
- Strict windowing without padding would give 49 frames (8000-400)/160+1 — see
  the numpy fallback, which uses that convention

Why mel-spectrograms?
1. Compact representation (8000 samples → 51×128 = 6528 values)
2. Perceptually meaningful (matches human hearing)
3. Works well with CNNs (treat as "image")
4. Standard in audio ML (proven approach)

Usage:
    processor = STFTProcessor(sample_rate=16000, n_mels=128)
    audio = np.array([...])  # (8000,) int16
    spectrogram = processor.process(audio)  # (51, 128)
"""

import logging
from typing import Optional, Tuple
import numpy as np

# We use numpy for basic STFT to minimize dependencies
# librosa can be used for more advanced features
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


class STFTProcessor:
    """
    STFT and Mel-spectrogram processor.

    Converts audio waveforms to mel-spectrograms suitable for
    input to the audio classification CNN.

    The processor can use either:
    1. librosa (if available) - more features, well-tested
    2. numpy-only fallback - fewer dependencies

    Attributes:
        sample_rate: Audio sample rate
        n_fft: FFT window size
        hop_length: Samples between successive frames
        n_mels: Number of mel filter banks
        window_length: Window size for STFT
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 160,
        n_mels: int = 128,
        window_length: int = 400,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        power: float = 2.0,
        normalize: bool = True
    ):
        """
        Initialize the STFT processor.

        Args:
            sample_rate: Audio sample rate in Hz
            n_fft: FFT size (should be >= window_length, power of 2)
            hop_length: Samples between STFT frames
            n_mels: Number of mel frequency bins
            window_length: Window size for STFT
            f_min: Minimum frequency for mel scale
            f_max: Maximum frequency (None = sample_rate/2)
            power: Exponent for magnitude spectrogram (2.0 = power spectrum)
            normalize: Whether to normalize output to [0, 1]
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.window_length = window_length
        self.f_min = f_min
        self.f_max = f_max if f_max else sample_rate / 2
        self.power = power
        self.normalize = normalize

        self.logger = logging.getLogger(__name__)

        # Create mel filter bank
        self._mel_filterbank = self._create_mel_filterbank()

        # Log configuration
        self.logger.info(
            f"STFT Processor initialized: "
            f"sr={sample_rate}, n_fft={n_fft}, hop={hop_length}, "
            f"n_mels={n_mels}, window={window_length}"
        )

    def _create_mel_filterbank(self) -> np.ndarray:
        """
        Create mel filter bank matrix.

        Returns:
            Filter bank of shape (n_mels, n_fft//2 + 1)
        """
        if LIBROSA_AVAILABLE:
            return librosa.filters.mel(
                sr=self.sample_rate,
                n_fft=self.n_fft,
                n_mels=self.n_mels,
                fmin=self.f_min,
                fmax=self.f_max
            )

        # Numpy fallback: Create mel filter bank manually
        return self._mel_filterbank_numpy()

    def _mel_filterbank_numpy(self) -> np.ndarray:
        """
        Create mel filter bank using only numpy.

        Based on the mel scale formula:
        mel = 2595 * log10(1 + f/700)
        f = 700 * (10^(mel/2595) - 1)
        """
        # Frequency bins
        n_freq = self.n_fft // 2 + 1
        freqs = np.linspace(0, self.sample_rate / 2, n_freq)

        # Mel scale conversion
        def hz_to_mel(hz):
            return 2595 * np.log10(1 + hz / 700)

        def mel_to_hz(mel):
            return 700 * (10 ** (mel / 2595) - 1)

        # Mel points
        mel_min = hz_to_mel(self.f_min)
        mel_max = hz_to_mel(self.f_max)
        mel_points = np.linspace(mel_min, mel_max, self.n_mels + 2)
        hz_points = mel_to_hz(mel_points)

        # Create triangular filters
        filterbank = np.zeros((self.n_mels, n_freq))

        for i in range(self.n_mels):
            # Lower, center, upper frequencies
            f_low = hz_points[i]
            f_center = hz_points[i + 1]
            f_high = hz_points[i + 2]

            # Rising edge
            rising = (freqs - f_low) / (f_center - f_low + 1e-10)
            # Falling edge
            falling = (f_high - freqs) / (f_high - f_center + 1e-10)

            filterbank[i] = np.maximum(0, np.minimum(rising, falling))

        return filterbank

    def process(self, audio: np.ndarray) -> np.ndarray:
        """
        Convert audio waveform to mel-spectrogram.

        Args:
            audio: Audio samples, shape (n_samples,), any numeric dtype

        Returns:
            Mel-spectrogram of shape (n_frames, n_mels)
            Values normalized to [0, 1] if normalize=True
        """
        # Convert to float32 and normalize to [-1, 1]
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        else:
            audio = audio.astype(np.float32)

        if LIBROSA_AVAILABLE:
            return self._process_librosa(audio)
        else:
            return self._process_numpy(audio)

    def _process_librosa(self, audio: np.ndarray) -> np.ndarray:
        """Process using librosa (preferred method)."""
        # Compute mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.window_length,
            n_mels=self.n_mels,
            fmin=self.f_min,
            fmax=self.f_max,
            power=self.power
        )

        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Transpose to (time, frequency)
        mel_spec_db = mel_spec_db.T

        # Normalize to [0, 1]
        if self.normalize:
            mel_spec_db = self._normalize(mel_spec_db)

        return mel_spec_db

    def _process_numpy(self, audio: np.ndarray) -> np.ndarray:
        """Process using only numpy (fallback method)."""
        # Compute STFT
        stft = self._stft_numpy(audio)

        # Compute power spectrogram
        power_spec = np.abs(stft) ** self.power

        # Apply mel filterbank
        mel_spec = np.dot(self._mel_filterbank, power_spec)

        # Convert to log scale
        mel_spec_db = 10 * np.log10(mel_spec + 1e-10)

        # Transpose to (time, frequency)
        mel_spec_db = mel_spec_db.T

        # Normalize
        if self.normalize:
            mel_spec_db = self._normalize(mel_spec_db)

        return mel_spec_db

    def _stft_numpy(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute STFT using numpy.

        Returns:
            Complex STFT matrix of shape (n_freq, n_frames)
        """
        # Pad audio
        n_samples = len(audio)
        n_frames = 1 + (n_samples - self.window_length) // self.hop_length

        # Create window
        window = np.hanning(self.window_length)

        # Initialize output
        n_freq = self.n_fft // 2 + 1
        stft = np.zeros((n_freq, n_frames), dtype=np.complex64)

        for i in range(n_frames):
            start = i * self.hop_length
            end = start + self.window_length

            # Extract and window frame
            frame = audio[start:end] * window

            # Pad to n_fft
            if len(frame) < self.n_fft:
                frame = np.pad(frame, (0, self.n_fft - len(frame)))

            # FFT
            spectrum = np.fft.rfft(frame)
            stft[:, i] = spectrum

        return stft

    def _normalize(self, spectrogram: np.ndarray) -> np.ndarray:
        """Normalize spectrogram to [0, 1] range."""
        min_val = spectrogram.min()
        max_val = spectrogram.max()

        if max_val - min_val > 1e-10:
            return (spectrogram - min_val) / (max_val - min_val)
        else:
            return np.zeros_like(spectrogram)

    def get_output_shape(self, n_samples: int) -> Tuple[int, int]:
        """
        Calculate output shape for given input length.

        Args:
            n_samples: Number of input audio samples

        Returns:
            Tuple of (n_frames, n_mels)
        """
        n_frames = 1 + (n_samples - self.window_length) // self.hop_length
        return (n_frames, self.n_mels)

    def process_batch(self, audio_batch: np.ndarray) -> np.ndarray:
        """
        Process a batch of audio samples.

        Args:
            audio_batch: Shape (batch_size, n_samples)

        Returns:
            Shape (batch_size, n_frames, n_mels)
        """
        return np.array([self.process(audio) for audio in audio_batch])
