"""
Audio CNN Model
===============

CNN architecture for mel-spectrogram classification.

Input: Mel-spectrogram (51, 128, 1)
    - 49 time frames (500ms audio)
    - 128 mel frequency bins
    - 1 channel

Architecture:
    3× Conv2D blocks with BatchNorm and MaxPool
    Global Average Pooling
    Dense layers with dropout

Total parameters: ~500K (fits on Raspberry Pi)
"""

import logging
from typing import Tuple, Optional

# TensorFlow
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    tf = None


def AudioCNN(
    input_shape: Tuple[int, int, int] = (51, 128, 1),
    num_classes: int = 10,
    dropout_rate: float = 0.3
):
    """
    Create Audio CNN model.

    Args:
        input_shape: Input spectrogram shape (time, freq, channels)
        num_classes: Number of output classes
        dropout_rate: Dropout probability

    Returns:
        Compiled Keras model
    """
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow not available")

    model = keras.Sequential([
        # Input
        layers.Input(shape=input_shape),

        # Block 1: 32 filters
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(dropout_rate / 2),

        # Block 2: 64 filters
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(dropout_rate / 2),

        # Block 3: 128 filters
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(dropout_rate / 2),

        # Global pooling
        layers.GlobalAveragePooling2D(),

        # Dense layers
        layers.Dense(128, activation='relu'),
        layers.Dropout(dropout_rate),

        # Output
        layers.Dense(num_classes, activation='softmax')
    ], name='audio_cnn')

    return model


def AudioCNNFeatureExtractor(
    input_shape: Tuple[int, int, int] = (51, 128, 1),
    output_dim: int = 128
):
    """
    Create Audio CNN feature extractor (for fusion model).

    Returns features without classification head.

    Args:
        input_shape: Input spectrogram shape
        output_dim: Output feature dimension

    Returns:
        Keras model outputting feature vector
    """
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow not available")

    model = keras.Sequential([
        layers.Input(shape=input_shape),

        # Conv blocks
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),

        # Feature output
        layers.GlobalAveragePooling2D(),
        layers.Dense(output_dim, activation='relu')
    ], name='audio_feature_extractor')

    return model


# Standalone test
if __name__ == "__main__":
    if TF_AVAILABLE:
        model = AudioCNN(num_classes=10)
        model.summary()
        print(f"\nTotal parameters: {model.count_params():,}")
