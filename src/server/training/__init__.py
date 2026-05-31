"""
Server Training Module
======================

Handles centralized training for Strategy A and B.

In centralized training:
1. Server receives data from edge nodes (via MQTT)
2. For Strategy A: Raw audio + labels → STFT → Training
3. For Strategy B: Spectrograms + labels → Training
4. Model is updated and broadcast to edge nodes

Classes:
    CentralizedTrainer: Main training orchestrator
    DataReceiver: Handles incoming data from edges
    ModelBroadcaster: Sends updated model to edges

Submodules:
    models/: Neural network architectures
"""

from .centralized_trainer import CentralizedTrainer

__all__ = ["CentralizedTrainer"]
