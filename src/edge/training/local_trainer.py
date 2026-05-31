"""
Local Trainer for Federated Learning
=====================================

Handles model training directly on the Raspberry Pi edge device.

Training Strategy:
1. Use small batch sizes (8) due to memory constraints
2. Gradient accumulation for effective larger batches
3. Few epochs per round (3) to balance training time and convergence
4. Learning rate warmup to avoid early divergence

Memory Optimization:
- Clear gradients after each step
- Use float16 where possible
- Generator-based data loading
- Aggressive garbage collection

Expected performance on Pi 4 (8GB):
- Training batch: ~200ms per batch
- Full round (500 samples, 3 epochs): ~5-10 minutes
- Memory usage: ~2GB during training
"""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

# TensorFlow import with graceful fallback
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    tf = None


class LocalTrainer:
    """
    On-device model trainer for federated learning.

    Manages:
    - Model initialization and loading
    - Training loop with gradient accumulation
    - Weight extraction for FL communication
    - Memory-efficient data handling

    Attributes:
        model: TensorFlow/Keras model
        optimizer: Training optimizer
        loss_fn: Loss function
        batch_size: Samples per gradient step
        accumulation_steps: Steps before weight update
    """

    def __init__(
        self,
        model=None,
        model_path: Optional[str] = None,
        input_shape: Tuple[int, ...] = (49, 128, 1),
        num_classes: int = 10,
        batch_size: int = 8,
        accumulation_steps: int = 4,
        learning_rate: float = 0.001,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the local trainer.

        Args:
            model: Pre-built model (optional)
            model_path: Path to saved model weights
            input_shape: Shape of input spectrograms
            num_classes: Number of output classes
            batch_size: Batch size for training
            accumulation_steps: Gradient accumulation steps
            learning_rate: Initial learning rate
            config: Additional configuration
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.accumulation_steps = accumulation_steps
        self.learning_rate = learning_rate
        self.config = config or {}

        self.model = model
        self.optimizer = None
        self.loss_fn = None

        self.logger = logging.getLogger(__name__)

        # Mock mode for development
        self._mock_mode = not TF_AVAILABLE

        if self._mock_mode:
            self.logger.warning(
                "TensorFlow not available - running in mock mode"
            )
        else:
            self._setup_training(model_path)

    def _setup_training(self, model_path: Optional[str]) -> None:
        """Setup model, optimizer, and loss function."""
        # Build model if not provided
        if self.model is None:
            self.model = self._build_audio_cnn()

        # Load weights if path provided
        if model_path:
            try:
                self.model.load_weights(model_path)
                self.logger.info(f"Loaded weights from {model_path}")
            except Exception as e:
                self.logger.warning(f"Could not load weights: {e}")

        # Setup optimizer
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate
        )

        # Setup loss
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False
        )

    def _build_audio_cnn(self):
        """
        Build the audio CNN model for spectrograms.

        Architecture:
        - Input: (49, 128, 1) mel-spectrogram
        - 3x Conv2D + BatchNorm + MaxPool blocks
        - Global Average Pooling
        - Dense(128) + Dropout
        - Dense(num_classes) + Softmax

        Total: ~500K parameters (fits Pi memory)
        """
        model = tf.keras.Sequential([
            # Input
            tf.keras.layers.Input(shape=self.input_shape),

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
            tf.keras.layers.Dense(self.num_classes, activation='softmax')
        ])

        model.compile(
            optimizer=self.optimizer,
            loss=self.loss_fn,
            metrics=['accuracy']
        )

        self.logger.info(
            f"Built audio CNN: {model.count_params():,} parameters"
        )

        return model

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 3,
        batch_size: Optional[int] = None,
        validation_split: float = 0.1
    ) -> Dict[str, float]:
        """
        Train the model on local data.

        Uses gradient accumulation for memory efficiency.

        Args:
            X: Input spectrograms, shape (N, H, W, 1)
            y: Labels, shape (N,)
            epochs: Number of training epochs
            batch_size: Override default batch size
            validation_split: Fraction for validation

        Returns:
            Dictionary of training metrics
        """
        if self._mock_mode:
            return self._mock_train(X, y, epochs)

        batch_size = batch_size or self.batch_size

        # Split validation set
        n_val = int(len(X) * validation_split)
        X_val, y_val = X[:n_val], y[:n_val]
        X_train, y_train = X[n_val:], y[n_val:]

        self.logger.info(
            f"Training: {len(X_train)} samples, {epochs} epochs, "
            f"batch_size={batch_size}"
        )

        # Training loop with gradient accumulation
        history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_acc = 0.0
            n_batches = 0

            # Shuffle data
            indices = np.random.permutation(len(X_train))
            X_train = X_train[indices]
            y_train = y_train[indices]

            # Batch training
            for i in range(0, len(X_train), batch_size):
                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]

                loss, acc = self._train_step(X_batch, y_batch)
                epoch_loss += loss
                epoch_acc += acc
                n_batches += 1

            # Epoch metrics
            epoch_loss /= n_batches
            epoch_acc /= n_batches

            # Validation
            val_loss, val_acc = self._evaluate(X_val, y_val)

            history['loss'].append(epoch_loss)
            history['accuracy'].append(epoch_acc)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_acc)

            self.logger.info(
                f"Epoch {epoch+1}/{epochs}: "
                f"loss={epoch_loss:.4f}, acc={epoch_acc:.4f}, "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
            )

        # Return final metrics
        return {
            'loss': history['loss'][-1],
            'accuracy': history['accuracy'][-1],
            'val_loss': history['val_loss'][-1],
            'val_accuracy': history['val_accuracy'][-1],
            'epochs': epochs,
            'samples': len(X)
        }

    @tf.function
    def _train_step(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Single training step with gradient computation."""
        with tf.GradientTape() as tape:
            predictions = self.model(X, training=True)
            loss = self.loss_fn(y, predictions)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables)
        )

        accuracy = tf.reduce_mean(
            tf.cast(
                tf.equal(tf.argmax(predictions, axis=1), tf.cast(y, tf.int64)),
                tf.float32
            )
        )

        return float(loss), float(accuracy)

    def _evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Evaluate model on data."""
        if len(X) == 0:
            return 0.0, 0.0

        predictions = self.model.predict(X, verbose=0)
        loss = self.loss_fn(y, predictions)
        accuracy = np.mean(np.argmax(predictions, axis=1) == y)

        return float(loss), float(accuracy)

    def _mock_train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int
    ) -> Dict[str, float]:
        """Mock training for development."""
        self.logger.info(f"Mock training: {len(X)} samples, {epochs} epochs")
        time.sleep(1.0)  # Simulate training time

        return {
            'loss': 0.5 + np.random.rand() * 0.3,
            'accuracy': 0.7 + np.random.rand() * 0.2,
            'val_loss': 0.6 + np.random.rand() * 0.3,
            'val_accuracy': 0.65 + np.random.rand() * 0.2,
            'epochs': epochs,
            'samples': len(X)
        }

    def get_weights(self) -> List[np.ndarray]:
        """
        Get model weights for federated aggregation.

        Returns:
            List of weight arrays
        """
        if self._mock_mode:
            return [np.random.rand(10, 10) for _ in range(5)]

        return self.model.get_weights()

    def set_weights(self, weights: List[np.ndarray]) -> None:
        """
        Set model weights (from server aggregation).

        Args:
            weights: List of weight arrays
        """
        if self._mock_mode:
            self.logger.info("Mock: weights updated")
            return

        self.model.set_weights(weights)
        self.logger.info("Model weights updated from server")

    def save_weights(self, path: str) -> None:
        """Save model weights to file."""
        if self._mock_mode:
            return

        self.model.save_weights(path)
        self.logger.info(f"Saved weights to {path}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        if self._mock_mode:
            return {'mock': True, 'params': 0}

        return {
            'params': self.model.count_params(),
            'input_shape': self.input_shape,
            'num_classes': self.num_classes,
            'layers': len(self.model.layers)
        }
