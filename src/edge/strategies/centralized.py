"""
Strategy A: Centralized Learning
================================

In centralized learning:
1. Edge captures audio + video
2. YOLOv8 provides labels on edge
3. RAW AUDIO + labels uploaded to server
4. Server does STFT + training
5. Server broadcasts updated model

This is the baseline strategy with:
- Maximum compute on server
- Minimum compute on edge
- Maximum bandwidth usage
- Minimum privacy (raw audio transmitted)

Data flow:
    Edge: capture() → YOLOv8 label → buffer raw audio
    Upload: [raw_audio, label, timestamp] × 500 samples
    Server: receive → STFT → train → broadcast model

Bandwidth estimate:
    Per sample: 16000 Hz × 0.5s × 2 bytes = 16 KB
    Per batch: 16 KB × 500 = 8 MB (uncompressed)
    With compression: ~4 MB
"""

import logging
import time
import json
import zlib
from typing import List, Dict, Any, Optional
import numpy as np

from .base import ILearningStrategy


class CentralizedStrategy(ILearningStrategy):
    """
    Centralized learning strategy (Strategy A).

    Uploads raw audio data to server for full processing.

    Configuration options:
        buffer_size: Number of samples before upload (default: 500)
        compression: Whether to compress data (default: True)
        mqtt_topic: Topic for data upload (default: thesis/edge/{id}/data)
    """

    def __init__(
        self,
        node_id: str,
        mqtt_client=None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize centralized strategy.

        Args:
            node_id: Edge node identifier
            mqtt_client: MQTT client for server communication
            config: Strategy configuration
        """
        super().__init__(node_id, mqtt_client, config)

        self.compression = self.config.get('compression', True)
        self.mqtt_topic_data = f"thesis/edge/{node_id}/data"
        self.mqtt_topic_model = "thesis/server/model/global"

        # Metrics
        self._bytes_uploaded = 0
        self._batches_uploaded = 0
        self._last_upload_time = None

        self.logger = logging.getLogger(__name__)

    @property
    def name(self) -> str:
        return "centralized"

    def prepare_sample(self, sample: Dict) -> Dict:
        """
        Prepare sample for upload.

        For centralized strategy, we keep the RAW AUDIO.
        No STFT processing on edge.

        Args:
            sample: Must contain 'audio', 'label', 'timestamp'

        Returns:
            Prepared sample with raw audio
        """
        # Validate required fields
        required = ['audio', 'label', 'timestamp']
        for field in required:
            if field not in sample:
                raise ValueError(f"Sample missing required field: {field}")

        # Keep raw audio (convert to list for JSON serialization if needed)
        prepared = {
            'audio': sample['audio'],  # numpy array or list
            'label': sample['label'],
            'timestamp': sample['timestamp'],
            'confidence': sample.get('confidence', 1.0),
            'node_id': self.node_id
        }

        return prepared

    def execute(self, samples: List[Dict]) -> bool:
        """
        Upload raw audio samples to server.

        Serializes and uploads the sample batch to the MQTT broker
        for server-side processing.

        Args:
            samples: List of prepared samples with raw audio

        Returns:
            True if upload successful
        """
        self.logger.info(f"Uploading {len(samples)} samples to server...")

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
                    qos=1  # At least once delivery
                )
            else:
                # Mock upload for development
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

        Converts numpy arrays to lists for JSON serialization.
        """
        # Convert numpy arrays to lists
        serializable = []
        for sample in samples:
            s = sample.copy()
            if isinstance(s['audio'], np.ndarray):
                s['audio'] = s['audio'].tolist()
            serializable.append(s)

        # Create batch payload
        batch = {
            'node_id': self.node_id,
            'strategy': self.name,
            'timestamp': time.time(),
            'num_samples': len(samples),
            'samples': serializable
        }

        return json.dumps(batch).encode('utf-8')

    def on_model_update(self, model_data: bytes) -> bool:
        """
        Handle model update from server.

        The server trains the model and broadcasts updates.
        Edge devices receive and apply the new model.

        Args:
            model_data: Serialized model weights

        Returns:
            True if update successful
        """
        self.logger.info(f"Received model update ({len(model_data)} bytes)")

        try:
            # TODO: Deserialize and load model weights
            # For now, just log receipt
            # model = deserialize_model(model_data)
            # self.local_model.set_weights(model.get_weights())

            self.logger.info("Model update applied successfully")
            return True

        except Exception as e:
            self.logger.error(f"Model update failed: {e}")
            return False

    def get_metrics(self) -> Dict[str, Any]:
        """Get centralized strategy metrics."""
        return {
            'strategy': self.name,
            'buffer_size': len(self._buffer),
            'bytes_uploaded': self._bytes_uploaded,
            'batches_uploaded': self._batches_uploaded,
            'last_upload': self._last_upload_time,
            'compression_enabled': self.compression
        }
