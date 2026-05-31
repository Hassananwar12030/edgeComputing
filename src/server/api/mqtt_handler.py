"""
MQTT Handler Module
===================

Handles MQTT communication on the server side.

Responsibilities:
1. Subscribe to edge data topics
2. Deserialize incoming data
3. Pass to trainer/aggregator
4. Publish model updates

Topic patterns:
    thesis/edge/+/data    - Training data (wildcard for node ID)
    thesis/edge/+/weights - Model weights
    thesis/edge/+/metrics - Edge metrics
    thesis/server/model/global - Global model (published)
    thesis/server/commands - Control commands (published)
"""

import logging
import json
import zlib
import threading
from typing import Callable, Dict, Any, Optional
import numpy as np

# MQTT client
try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    mqtt = None


class MQTTHandler:
    """
    Server-side MQTT handler.

    Manages subscriptions to edge topics and publishes
    model updates.

    Attributes:
        broker: MQTT broker address
        port: MQTT broker port
        trainer: Reference to trainer for data callback
    """

    def __init__(
        self,
        broker: str = "localhost",
        port: int = 1883,
        trainer=None,
        aggregator=None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        use_tls: bool = False
    ):
        """
        Initialize MQTT handler.

        Args:
            broker: MQTT broker hostname
            port: MQTT broker port
            trainer: CentralizedTrainer instance
            aggregator: FedAvgAggregator instance
            username: MQTT username
            password: MQTT password
            use_tls: Enable TLS
        """
        self.broker = broker
        self.port = port
        self.trainer = trainer
        self.aggregator = aggregator
        self.username = username
        self.password = password
        self.use_tls = use_tls

        self.client = None
        self.is_connected = False

        self.logger = logging.getLogger(__name__)

        # Callbacks for custom handlers
        self._data_callback: Optional[Callable] = None
        self._weights_callback: Optional[Callable] = None
        self._metrics_callback: Optional[Callable] = None

        if not MQTT_AVAILABLE:
            self.logger.warning("paho-mqtt not available")

    def start(self) -> bool:
        """
        Start MQTT handler.

        Connects to broker and subscribes to edge topics.
        """
        if not MQTT_AVAILABLE:
            self.logger.error("Cannot start - MQTT not available")
            return False

        try:
            # Create client
            client_id = f"thesis_server_{id(self)}"
            self.client = mqtt.Client(client_id=client_id)

            # Set callbacks
            self.client.on_connect = self._on_connect
            self.client.on_disconnect = self._on_disconnect
            self.client.on_message = self._on_message

            # Authentication
            if self.username and self.password:
                self.client.username_pw_set(self.username, self.password)

            # Connect
            self.client.connect(self.broker, self.port, keepalive=60)
            self.client.loop_start()

            self.logger.info(
                f"MQTT handler connecting to {self.broker}:{self.port}"
            )
            return True

        except Exception as e:
            self.logger.error(f"MQTT start failed: {e}")
            return False

    def stop(self) -> None:
        """Stop MQTT handler."""
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
            self.is_connected = False
            self.logger.info("MQTT handler stopped")

    def _on_connect(self, client, userdata, flags, rc):
        """Handle connection event."""
        if rc == 0:
            self.is_connected = True
            self.logger.info("Connected to MQTT broker")

            # Subscribe to edge topics
            client.subscribe("thesis/edge/+/data", qos=1)
            client.subscribe("thesis/edge/+/weights", qos=2)
            client.subscribe("thesis/edge/+/metrics", qos=0)

            self.logger.info("Subscribed to edge topics")
        else:
            self.logger.error(f"Connection failed: rc={rc}")

    def _on_disconnect(self, client, userdata, rc):
        """Handle disconnection event."""
        self.is_connected = False
        self.logger.warning(f"Disconnected from MQTT broker: rc={rc}")

    def _on_message(self, client, userdata, message):
        """Handle incoming message."""
        topic = message.topic
        payload = message.payload

        self.logger.debug(f"Received message on {topic}: {len(payload)} bytes")

        try:
            # Route based on topic
            if "/data" in topic:
                self._handle_data(topic, payload)
            elif "/weights" in topic:
                self._handle_weights(topic, payload)
            elif "/metrics" in topic:
                self._handle_metrics(topic, payload)
            else:
                self.logger.warning(f"Unknown topic: {topic}")

        except Exception as e:
            self.logger.error(f"Message handling error: {e}")

    def _handle_data(self, topic: str, payload: bytes) -> None:
        """
        Handle incoming training data.

        Args:
            topic: MQTT topic
            payload: Message payload (possibly compressed)
        """
        # Extract node ID from topic
        parts = topic.split('/')
        node_id = parts[2] if len(parts) > 2 else "unknown"

        # Decompress if needed
        try:
            payload = zlib.decompress(payload)
        except zlib.error:
            pass  # Not compressed

        # Parse data
        try:
            # Try numpy format first
            import io
            data = np.load(io.BytesIO(payload), allow_pickle=True)

            spectrograms = data['spectrograms']
            metadata = json.loads(str(data['metadata']))
            labels = [m['label'] for m in metadata]

            self.logger.info(
                f"Received {len(spectrograms)} samples from {node_id}"
            )

            # Pass to trainer
            if self.trainer:
                self.trainer.add_batch(spectrograms, labels, node_id)

            # Custom callback
            if self._data_callback:
                self._data_callback(node_id, spectrograms, labels)

        except Exception as e:
            # Try JSON format
            try:
                batch = json.loads(payload.decode('utf-8'))
                samples = batch.get('samples', [])
                labels = [s['label'] for s in samples]

                self.logger.info(
                    f"Received {len(samples)} JSON samples from {node_id}"
                )

                if self._data_callback:
                    self._data_callback(node_id, samples, labels)

            except Exception as e2:
                self.logger.error(f"Failed to parse data: {e}, {e2}")

    def _handle_weights(self, topic: str, payload: bytes) -> None:
        """Handle incoming model weights."""
        parts = topic.split('/')
        node_id = parts[2] if len(parts) > 2 else "unknown"

        self.logger.info(f"Received weights from {node_id}")

        # TODO: Deserialize weights and pass to aggregator
        if self._weights_callback:
            self._weights_callback(node_id, payload)

    def _handle_metrics(self, topic: str, payload: bytes) -> None:
        """Handle incoming metrics."""
        parts = topic.split('/')
        node_id = parts[2] if len(parts) > 2 else "unknown"

        try:
            metrics = json.loads(payload.decode('utf-8'))
            self.logger.debug(f"Metrics from {node_id}: {metrics}")

            if self._metrics_callback:
                self._metrics_callback(node_id, metrics)

        except Exception as e:
            self.logger.error(f"Failed to parse metrics: {e}")

    def publish_model(self, model_bytes: bytes) -> bool:
        """
        Publish updated model to all edges.

        Args:
            model_bytes: Serialized model

        Returns:
            True if published successfully
        """
        if not self.is_connected:
            self.logger.warning("Not connected - cannot publish model")
            return False

        topic = "thesis/server/model/global"
        result = self.client.publish(topic, model_bytes, qos=2, retain=True)

        if result.rc == 0:
            self.logger.info(f"Published model ({len(model_bytes)} bytes)")
            return True
        else:
            self.logger.error(f"Model publish failed: rc={result.rc}")
            return False

    def publish_command(self, command: Dict[str, Any]) -> bool:
        """Publish command to all edges."""
        if not self.is_connected:
            return False

        topic = "thesis/server/commands"
        payload = json.dumps(command).encode('utf-8')
        result = self.client.publish(topic, payload, qos=1)

        return result.rc == 0

    def set_data_callback(self, callback: Callable) -> None:
        """Set callback for incoming data."""
        self._data_callback = callback

    def set_weights_callback(self, callback: Callable) -> None:
        """Set callback for incoming weights."""
        self._weights_callback = callback

    def set_metrics_callback(self, callback: Callable) -> None:
        """Set callback for incoming metrics."""
        self._metrics_callback = callback
