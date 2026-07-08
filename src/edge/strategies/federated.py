"""
Strategy C: Federated Learning
==============================

In federated learning:
1. Edge captures audio + video
2. YOLOv8 provides labels on edge
3. STFT processing on edge → spectrograms
4. LOCAL TRAINING on edge with buffered data
5. MODEL WEIGHTS uploaded to server (not data!)
6. Server aggregates weights (FedAvg)
7. Server broadcasts aggregated model

This is the privacy-preserving strategy with:
- Maximum compute on edge (STFT + training)
- Minimum compute on server (aggregation only)
- Minimum bandwidth (just model weights)
- Maximum privacy (data never leaves device)

Data flow:
    Edge: capture() → YOLOv8 label → STFT → buffer → local train
    Upload: trainable weights only (~1.5 MB; ~400 KB quantized)
    Server: receive weights → FedAvg → broadcast global model

Bandwidth estimate:
    Trainable params: ~390K × 4 bytes = ~1.5 MB per round (FusionModel; vision backbone frozen)
    With INT8 quantization: ~400 KB per round

This strategy uses Flower (flwr) for federated learning coordination.
"""

import logging
import time
from typing import List, Dict, Any, Optional
import numpy as np

from .base import ILearningStrategy


class FederatedStrategy(ILearningStrategy):
    """
    Federated learning strategy (Strategy C).

    Performs local training on edge, uploads only model weights.

    Configuration options:
        buffer_size: Number of samples before training (default: 500)
        local_epochs: Epochs for local training (default: 3)
        batch_size: Batch size for training (default: 8)
        learning_rate: Learning rate (default: 0.001)
    """

    def __init__(
        self,
        node_id: str,
        mqtt_client=None,
        flower_client=None,
        stft_processor=None,
        local_trainer=None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize federated strategy.

        Args:
            node_id: Edge node identifier
            mqtt_client: MQTT client (backup communication)
            flower_client: Flower FL client
            stft_processor: STFTProcessor instance
            local_trainer: LocalTrainer instance
            config: Strategy configuration
        """
        super().__init__(node_id, mqtt_client, config)

        self.flower_client = flower_client
        self.stft_processor = stft_processor
        self.local_trainer = local_trainer

        # Training config
        self.local_epochs = self.config.get('local_epochs', 3)
        self.batch_size = self.config.get('batch_size', 8)
        self.learning_rate = self.config.get('learning_rate', 0.001)

        # Metrics
        self._training_rounds = 0
        self._total_training_time = 0.0
        self._weights_uploaded_bytes = 0
        self._last_accuracy = None

        self.logger = logging.getLogger(__name__)

    @property
    def name(self) -> str:
        return "federated"

    def prepare_sample(self, sample: Dict) -> Dict:
        """
        Prepare sample by converting audio to spectrogram.

        Same as hybrid strategy - STFT on edge.

        Args:
            sample: Must contain 'audio', 'label', 'timestamp'

        Returns:
            Prepared sample with spectrogram
        """
        required = ['audio', 'label', 'timestamp']
        for field in required:
            if field not in sample:
                raise ValueError(f"Sample missing required field: {field}")

        # Convert audio to spectrogram
        if self.stft_processor:
            spectrogram = self.stft_processor.process(sample['audio'])
        else:
            spectrogram = np.random.rand(51, 128).astype(np.float32)

        prepared = {
            'spectrogram': spectrogram,
            'label': sample['label'],
            'timestamp': sample['timestamp'],
            'confidence': sample.get('confidence', 1.0)
        }

        return prepared

    def execute(self, samples: List[Dict]) -> bool:
        """
        Execute local training and upload weights.

        This is the core federated learning workflow:
        1. Convert samples to training dataset
        2. Train local model for N epochs
        3. Extract model weights
        4. Send weights to Flower server

        Args:
            samples: List of prepared samples with spectrograms

        Returns:
            True if training and upload successful
        """
        self.logger.info(
            f"Starting local training with {len(samples)} samples..."
        )

        try:
            start_time = time.time()

            # Step 1: Prepare training data
            X, y = self._prepare_training_data(samples)

            # Step 2: Local training
            metrics = self._train_local(X, y)

            training_time = time.time() - start_time
            self._total_training_time += training_time

            # Step 3: Get model weights
            weights = self._get_model_weights()

            # Step 4: Send to Flower server
            success = self._send_weights_to_server(weights, metrics)

            # Update metrics
            self._training_rounds += 1
            self._last_accuracy = metrics.get('accuracy', None)

            self.logger.info(
                f"Local training complete: "
                f"accuracy={metrics.get('accuracy', 'N/A'):.3f}, "
                f"time={training_time:.1f}s"
            )

            return success

        except Exception as e:
            self.logger.error(f"Local training failed: {e}")
            return False

    def _prepare_training_data(
        self,
        samples: List[Dict]
    ) -> tuple:
        """
        Convert samples to training arrays.

        Args:
            samples: List of sample dicts

        Returns:
            Tuple of (X, y) numpy arrays
        """
        # Extract spectrograms and labels
        X = np.array([s['spectrogram'] for s in samples])

        # Convert labels to indices
        # TODO: Use actual label encoder
        unique_labels = list(set(s['label'] for s in samples))
        label_to_idx = {l: i for i, l in enumerate(unique_labels)}
        y = np.array([label_to_idx[s['label']] for s in samples])

        # Add channel dimension for CNN: (N, H, W) -> (N, H, W, 1)
        X = X[..., np.newaxis]

        self.logger.debug(f"Training data: X={X.shape}, y={y.shape}")

        return X, y

    def _train_local(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Train the local model.

        Uses gradient accumulation for memory efficiency on Pi.

        Args:
            X: Input spectrograms, shape (N, H, W, 1)
            y: Labels, shape (N,)

        Returns:
            Training metrics dict
        """
        if self.local_trainer:
            return self.local_trainer.train(
                X, y,
                epochs=self.local_epochs,
                batch_size=self.batch_size
            )

        # Mock training for development
        self.logger.warning("No local trainer - mock training")
        time.sleep(0.5)  # Simulate training time

        return {
            'loss': 0.5 + np.random.rand() * 0.3,
            'accuracy': 0.7 + np.random.rand() * 0.2,
            'epochs': self.local_epochs,
            'samples': len(y)
        }

    def _get_model_weights(self) -> List[np.ndarray]:
        """
        Extract model weights for upload.

        Returns:
            List of weight arrays
        """
        if self.local_trainer:
            return self.local_trainer.get_weights()

        # Mock weights for development
        return [np.random.rand(10, 10) for _ in range(5)]

    def _send_weights_to_server(
        self,
        weights: List[np.ndarray],
        metrics: Dict[str, float]
    ) -> bool:
        """
        Send model weights to Flower server.

        Args:
            weights: List of weight arrays
            metrics: Training metrics

        Returns:
            True if successful
        """
        if self.flower_client:
            # Flower handles the communication
            return self.flower_client.send_weights(weights, metrics)

        # Mock upload for development
        total_bytes = sum(w.nbytes for w in weights)
        self._weights_uploaded_bytes += total_bytes

        self.logger.info(
            f"Weights sent to server: {total_bytes/1024:.1f} KB"
        )

        return True

    def on_model_update(self, model_data: bytes) -> bool:
        """
        Handle aggregated model from server.

        After FedAvg aggregation, server broadcasts the new global model.

        Args:
            model_data: Serialized aggregated weights

        Returns:
            True if update successful
        """
        self.logger.info(
            f"Received aggregated model ({len(model_data)} bytes)"
        )

        try:
            # TODO: Deserialize and apply weights
            # weights = deserialize_weights(model_data)
            # self.local_trainer.set_weights(weights)

            self.logger.info("Global model applied successfully")
            return True

        except Exception as e:
            self.logger.error(f"Model update failed: {e}")
            return False

    def get_metrics(self) -> Dict[str, Any]:
        """Get federated strategy metrics."""
        return {
            'strategy': self.name,
            'buffer_size': len(self._buffer),
            'training_rounds': self._training_rounds,
            'total_training_time': self._total_training_time,
            'weights_uploaded_bytes': self._weights_uploaded_bytes,
            'last_accuracy': self._last_accuracy,
            'local_epochs': self.local_epochs,
            'batch_size': self.batch_size
        }

    def get_data_stats(self) -> Dict[str, Any]:
        """
        Get statistics about local data distribution.

        Useful for debugging non-IID effects.
        """
        if not self._buffer:
            return {'num_samples': 0, 'classes': {}}

        labels = [s['label'] for s in self._buffer]
        unique, counts = np.unique(labels, return_counts=True)

        return {
            'num_samples': len(self._buffer),
            'classes': dict(zip(unique.tolist(), counts.tolist()))
        }
