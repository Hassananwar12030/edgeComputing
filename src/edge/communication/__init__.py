"""
Communication Module
====================

Handles all network communication from edge devices:
- MQTT client for message passing (all strategies)
- Flower client for federated learning (Strategy C)

MQTT Topics:
    thesis/edge/{id}/data     - Data uploads (Strategy A/B)
    thesis/edge/{id}/weights  - Model weights (Strategy C)
    thesis/edge/{id}/metrics  - Performance metrics
    thesis/server/model/global - Model updates from server
    thesis/server/commands    - Control commands

Classes:
    MQTTClient: MQTT publish/subscribe wrapper
    FlowerClient: Flower FL client wrapper
"""

from .mqtt_client import MQTTClient
from .flower_client import FlowerClient

__all__ = ["MQTTClient", "FlowerClient"]
