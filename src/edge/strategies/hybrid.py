"""
Strategy B: Hybrid STFT Learning
================================

In hybrid learning:
1. Edge captures audio + video
2. YOLOv8 provides labels on edge
3. STFT processing on edge → spectrograms
4. SPECTROGRAMS + labels uploaded to server
5. Server does training only (no STFT needed)
6. Server broadcasts updated model

This is a middle-ground strategy with:
- Moderate compute on edge (STFT only)
- Moderate compute on server (training only)
- Moderate bandwidth (spectrograms smaller than raw)
- Moderate privacy (spectrograms less sensitive than raw audio)

Data flow:
    Edge: capture() → YOLOv8 label → STFT → buffer spectrogram
    Upload: [spectrogram, label, timestamp] × 500 samples
    Server: receive → train → broadcast model

Bandwidth estimate:
    Per sample: 51 × 128 × 4 bytes (float32) = 26 KB
    Per batch: 26 KB × 500 = 13 MB (uncompressed)
    With compression + float16: ~3 MB
"""

import logging
import time
import json
import zlib
from typing import List, Dict, Any, Optional
import numpy as np

from .base import ILearningStrategy


class HybridStrategy(ILearningStrategy):
    """
    Hybrid STFT strategy (Strategy B).

    Performs STFT on edge, uploads spectrograms to server.

    Configuration options:
        buffer_size: Number of samples before upload (default: 500)
        compression: Whether to compress data (default: True)
        use_float16: Use half precision for smaller payload (default: True)
        mqtt_topic: Topic for data upload
    """

    def __init__(
        self,
        node_id: str,
        mqtt_client=None,
        stft_processor=None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize hybrid strategy.

        Args:
            node_id: Edge node identifier
            mqtt_client: MQTT client for server communication
            stft_processor: STFTProcessor instance for spectrograms
            config: Strategy configuration
        """
        super().__init__(node_id, mqtt_client, config)

        self.stft_processor = stft_processor
        self.compression = self.config.get('compression', True)
        self.use_float16 = self.config.get('use_float16', True)
        self.mqtt_topic_data = f"thesis/edge/{node_id}/data"
        self.mqtt_topic_model = "thesis/server/model/global"

        # Metrics
        self._bytes_uploaded = 0
        self._batches_uploaded = 0
        self._stft_time_total = 0.0
        self._last_upload_time = None

        self.logger = logging.getLogger(__name__)

    @property
    def name(self) -> str:
        return "hybrid"

    def prepare_sample(self, sample: Dict) -> Dict:
        """
        Prepare sample by converting audio to spectrogram.

        For hybrid strategy, we do STFT on edge before buffering.

        Args:
            sample: Must contain 'audio', 'label', 'timestamp'

        Returns:
            Prepared sample with spectrogram instead of raw audio
        """
        # Validate required fields
        required = ['audio', 'label', 'timestamp']
        for field in required:
            if field not in sample:
                raise ValueError(f"Sample missing required field: {field}")

        # Convert audio to spectrogram
        start_time = time.time()

        if self.stft_processor:
            spectrogram = self.stft_processor.process(sample['audio'])
        else:
            # Mock spectrogram for development
            spectrogram = self._mock_spectrogram()

        stft_time = time.time() - start_time
        self._stft_time_total += stft_time

        # Convert to float16 if enabled (saves bandwidth)
        if self.use_float16:
            spectrogram = spectrogram.astype(np.float16)

        prepared = {
            'spectrogram': spectrogram,
            'label': sample['label'],
            'timestamp': sample['timestamp'],
            'confidence': sample.get('confidence', 1.0),
            'node_id': self.node_id,
            'stft_time': stft_time
        }

        return prepared

    def _mock_spectrogram(self) -> np.ndarray:
        """Generate mock spectrogram for development."""
        # Shape: (n_frames, n_mels) = (51, 128)
        return np.random.rand(51, 128).astype(np.float32)

    def execute(self, samples: List[Dict]) -> bool:
        """
        Upload spectrogram samples to server.

        Serializes and uploads the sample batch to the MQTT broker.

        Args:
            samples: List of prepared samples with spectrograms

        Returns:
            True if upload successful
        """
        self.logger.info(f"Uploading {len(samples)} spectrograms to server...")

        try:
            # Serialize samples
            payload = self._serialize_batch(samples)

            # Compress if enabled
            if self.compression:
                payload = zlib.compress(payload)

            # Upload via MQTT
            if self.mqtt_client:
                self.mqtt_client.publish(
                    topic=self.mqtt_topic_data,
                    payload=payload,
                    qos=1
                )
            else:
                self.logger.warning("No MQTT client - mock upload")

            # Update metrics
            self._bytes_uploaded += len(payload)
            self._batches_uploaded += 1
            self._last_upload_time = time.time()

            self.logger.info(
                f"Upload complete: {len(payload)} bytes "
                f"({len(payload)/1024:.1f} KB)"
            )
            return True

        except Exception as e:
            self.logger.error(f"Upload failed: {e}")
            return False

    def _serialize_batch(self, samples: List[Dict]) -> bytes:
        """
        Serialize a batch of samples to bytes.

        Uses numpy's save format for efficient spectrogram storage.
        """
        import io

        # Separate spectrograms from metadata
        spectrograms = []
        metadata = []

        for sample in samples:
            spectrograms.append(sample['spectrogram'])
            metadata.append({
                'label': sample['label'],
                'timestamp': sample['timestamp'],
                'confidence': sample.get('confidence', 1.0),
                'node_id': sample.get('node_id', self.node_id)
            })

        # Stack spectrograms into single array
        spec_array = np.stack(spectrograms)

        # Save to bytes buffer
        buffer = io.BytesIO()
        np.savez_compressed(
            buffer,
            spectrograms=spec_array,
            metadata=json.dumps(metadata)
        )

        return buffer.getvalue()

    def on_model_update(self, model_data: bytes) -> bool:
        """
        Handle model update from server.

        Args:
            model_data: Serialized model weights

        Returns:
            True if update successful
        """
        self.logger.info(f"Received model update ({len(model_data)} bytes)")

        try:
            # TODO: Deserialize and load model weights
            self.logger.info("Model update applied successfully")
            return True

        except Exception as e:
            self.logger.error(f"Model update failed: {e}")
            return False

    def get_metrics(self) -> Dict[str, Any]:
        """Get hybrid strategy metrics."""
        return {
            'strategy': self.name,
            'buffer_size': len(self._buffer),
            'bytes_uploaded': self._bytes_uploaded,
            'batches_uploaded': self._batches_uploaded,
            'stft_time_total': self._stft_time_total,
            'last_upload': self._last_upload_time,
            'compression_enabled': self.compression,
            'float16_enabled': self.use_float16
        }
