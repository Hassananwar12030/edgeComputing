"""
Centralized Trainer Module
==========================

Handles centralized model training for Strategy A and B.

Workflow:
1. Receive data batches from edge nodes (raw audio or spectrograms)
2. Preprocess if needed (STFT for Strategy A)
3. Add to training buffer
4. Train when enough data accumulated
5. Broadcast updated model

Training triggers:
- Time-based: Every N minutes
- Data-based: Every N samples
- Manual: On command

Usage:
    trainer = CentralizedTrainer(model, strategy="hybrid")
    trainer.add_batch(spectrograms, labels)  # From MQTT
    if trainer.should_train():
        trainer.train()
        model_bytes = trainer.get_model_bytes()
        mqtt.publish(model_bytes)
"""

import logging
import time
import threading
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

# TensorFlow
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    tf = None


class CentralizedTrainer:
    """
    Centralized training manager for server.

    Handles:
    - Data buffering from multiple edge nodes
    - STFT processing (for Strategy A)
    - Model training
    - Model serialization for broadcast

    Attributes:
        model: TensorFlow model
        strategy: "centralized" or "hybrid"
        buffer: Training data buffer
    """

    def __init__(
        self,
        model=None,
        strategy: str = "hybrid",
        buffer_size: int = 5000,
        batch_size: int = 32,
        epochs_per_train: int = 5,
        train_trigger_samples: int = 1000,
        stft_processor=None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize centralized trainer.

        Args:
            model: TensorFlow/Keras model (None to build default)
            strategy: "centralized" (raw audio) or "hybrid" (spectrograms)
            buffer_size: Maximum samples in buffer
            batch_size: Training batch size
            epochs_per_train: Epochs per training session
            train_trigger_samples: Samples before auto-train
            stft_processor: STFTProcessor for Strategy A
            config: Additional configuration
        """
        self.strategy = strategy
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.epochs_per_train = epochs_per_train
        self.train_trigger_samples = train_trigger_samples
        self.stft_processor = stft_processor
        self.config = config or {}

        self.model = model
        self.buffer_X: List[np.ndarray] = []
        self.buffer_y: List[int] = []
        self.label_encoder: Dict[str, int] = {}

        self._lock = threading.Lock()
        self._training_count = 0
        self._last_train_time = None

        self.logger = logging.getLogger(__name__)

        # Build model if not provided
        if self.model is None and TF_AVAILABLE:
            self.model = self._build_model()

    def _build_model(self):
        """Build the fusion model for training."""
        # Audio CNN (same as edge model)
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(49, 128, 1)),

            # Block 1
            tf.keras.layers.Conv2D(32, (3, 3), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),

            # Block 2
            tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),

            # Block 3
            tf.keras.layers.Conv2D(128, (3, 3), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),

            # Classification head
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(10, activation='softmax')  # 10 classes
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        self.logger.info(f"Built model: {model.count_params():,} parameters")
        return model

    def add_batch(
        self,
        data: np.ndarray,
        labels: List[str],
        node_id: str = "unknown"
    ) -> int:
        """
        Add a batch of data to the training buffer.

        Args:
            data: Spectrograms or raw audio (depending on strategy)
            labels: List of string labels
            node_id: Source edge node ID

        Returns:
            Number of samples added
        """
        with self._lock:
            # Process data if needed (Strategy A: raw audio → spectrogram)
            if self.strategy == "centralized" and self.stft_processor:
                processed = []
                for audio in data:
                    spec = self.stft_processor.process(audio)
                    processed.append(spec)
                data = np.array(processed)

            # Encode labels
            encoded_labels = []
            for label in labels:
                if label not in self.label_encoder:
                    self.label_encoder[label] = len(self.label_encoder)
                encoded_labels.append(self.label_encoder[label])

            # Add to buffer
            for i in range(len(data)):
                self.buffer_X.append(data[i])
                self.buffer_y.append(encoded_labels[i])

            # Enforce buffer size limit
            while len(self.buffer_X) > self.buffer_size:
                self.buffer_X.pop(0)
                self.buffer_y.pop(0)

            self.logger.info(
                f"Added {len(data)} samples from {node_id}. "
                f"Buffer: {len(self.buffer_X)}/{self.buffer_size}"
            )

            return len(data)

    def should_train(self) -> bool:
        """Check if training should be triggered."""
        return len(self.buffer_X) >= self.train_trigger_samples

    def train(self, epochs: Optional[int] = None) -> Dict[str, float]:
        """
        Train the model on buffered data.

        Args:
            epochs: Override default epochs

        Returns:
            Training metrics
        """
        if not TF_AVAILABLE:
            self.logger.warning("TensorFlow not available")
            return {}

        epochs = epochs or self.epochs_per_train

        with self._lock:
            if len(self.buffer_X) == 0:
                self.logger.warning("No data to train on")
                return {}

            # Prepare data
            X = np.array(self.buffer_X)
            y = np.array(self.buffer_y)

            # Add channel dimension if needed
            if X.ndim == 3:
                X = X[..., np.newaxis]

        self.logger.info(
            f"Training on {len(X)} samples for {epochs} epochs"
        )

        # Train
        start_time = time.time()

        history = self.model.fit(
            X, y,
            batch_size=self.batch_size,
            epochs=epochs,
            validation_split=0.1,
            verbose=1
        )

        train_time = time.time() - start_time

        # Get final metrics
        metrics = {
            'loss': float(history.history['loss'][-1]),
            'accuracy': float(history.history['accuracy'][-1]),
            'val_loss': float(history.history['val_loss'][-1]),
            'val_accuracy': float(history.history['val_accuracy'][-1]),
            'samples': len(X),
            'epochs': epochs,
            'train_time': train_time
        }

        self._training_count += 1
        self._last_train_time = time.time()

        self.logger.info(
            f"Training complete: "
            f"loss={metrics['loss']:.4f}, "
            f"accuracy={metrics['accuracy']:.4f}, "
            f"time={train_time:.1f}s"
        )

        return metrics

    def get_model_bytes(self) -> bytes:
        """
        Serialize model for broadcast.

        Returns:
            Serialized model weights
        """
        if not TF_AVAILABLE or self.model is None:
            return b''

        import io
        import pickle

        weights = self.model.get_weights()
        buffer = io.BytesIO()
        pickle.dump(weights, buffer)

        return buffer.getvalue()

    def load_model_bytes(self, model_bytes: bytes) -> bool:
        """
        Load model from bytes.

        Args:
            model_bytes: Serialized model

        Returns:
            True if successful
        """
        if not TF_AVAILABLE:
            return False

        try:
            import io
            import pickle

            buffer = io.BytesIO(model_bytes)
            weights = pickle.load(buffer)
            self.model.set_weights(weights)
            return True

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False

    def save_model(self, path: str) -> None:
        """Save model to file."""
        if self.model:
            self.model.save(path)
            self.logger.info(f"Model saved to {path}")

    def get_buffer_status(self) -> Dict[str, Any]:
        """Get buffer status."""
        return {
            'buffer_size': len(self.buffer_X),
            'max_size': self.buffer_size,
            'classes': len(self.label_encoder),
            'training_count': self._training_count,
            'last_train_time': self._last_train_time
        }

    def clear_buffer(self) -> int:
        """Clear the buffer."""
        with self._lock:
            count = len(self.buffer_X)
            self.buffer_X.clear()
            self.buffer_y.clear()
            return count
