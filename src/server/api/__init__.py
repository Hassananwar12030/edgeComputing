"""
Server API Module
=================

Handles communication with edge devices.

Components:
    MQTTHandler: Receives data/weights via MQTT
    DataProcessor: Processes incoming data batches
    ModelPublisher: Broadcasts model updates

Topics handled:
    thesis/edge/+/data    - Data uploads (Strategy A/B)
    thesis/edge/+/weights - Weight uploads (Strategy C)
    thesis/edge/+/metrics - Metrics from edges
"""

from .mqtt_handler import MQTTHandler

__all__ = ["MQTTHandler"]
