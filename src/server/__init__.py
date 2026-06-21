"""
Server Module - Runs on Cloud/Laptop
=====================================

This module contains all code that executes on the central server.
The server role differs by strategy:

Strategy A (Centralized):
    - Receives raw audio + labels
    - Performs STFT conversion
    - Trains model centrally
    - Broadcasts updated model

Strategy B (Hybrid):
    - Receives spectrograms + labels
    - Trains model centrally
    - Broadcasts updated model

Strategy C (Federated):
    - Runs Flower FL server
    - Receives model weights from clients
    - Performs FedAvg aggregation
    - Broadcasts global model

Submodules:
    - flower_server: Flower FL server (Strategy C)
    - training: Centralized training (Strategy A/B)
    - aggregation: FedAvg and other aggregation methods
    - api: MQTT message handling

Usage:
    # Start Flower server
    python -m src.server.flower_server

    # Start centralized training server
    python -m src.server.main --strategy centralized
"""

# Submodules are NOT eagerly imported — Flower / TF dependencies should only
# load when the user actually reaches for those features. Import explicitly:
#     from src.server.flower_server import FlowerServer
#     from src.server.main import Server
__all__ = []
