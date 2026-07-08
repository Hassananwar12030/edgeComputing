"""
MQTT Client Module
==================

Handles MQTT communication between edge devices and server.

MQTT (Message Queuing Telemetry Transport) is ideal for IoT:
- Lightweight protocol
- Supports QoS levels for reliability
- Pub/sub pattern for scalability
- TLS support for security

Topic structure:
    thesis/edge/{node_id}/data     - Training data (QoS 1)
    thesis/edge/{node_id}/weights  - FL weights (QoS 2)
    thesis/edge/{node_id}/metrics  - Metrics (QoS 0)
    thesis/server/model/global     - Global model (QoS 2)
    thesis/server/commands         - Commands (QoS 1)

Usage:
    client = MQTTClient(broker="192.168.1.100", node_id="A")
    client.connect()
    client.publish("thesis/edge/A/data", payload)
    client.subscribe("thesis/server/model/global", callback)
"""

import logging
import time
import json
import ssl
from typing import Callable, Optional, Any, Dict
from threading import Thread

# paho-mqtt is the standard MQTT client for Python
try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    mqtt = None


class MQTTClient:
    """
    MQTT client wrapper for edge-server communication.

    Provides simple publish/subscribe interface with:
    - Automatic reconnection
    - TLS support
    - Message callbacks
    - Connection monitoring

    Attributes:
        broker: MQTT broker address
        port: MQTT broker port
        node_id: This edge node's identifier
        client: paho MQTT client instance
    """

    def __init__(
        self,
        broker: str = "localhost",
        port: int = 1883,
        node_id: str = "edge_0",
        username: Optional[str] = None,
        password: Optional[str] = None,
        use_tls: bool = False,
        ca_cert: Optional[str] = None,
        client_cert: Optional[str] = None,
        client_key: Optional[str] = None
    ):
        """
        Initialize MQTT client.

        Args:
            broker: MQTT broker hostname/IP
            port: MQTT broker port (1883 or 8883 for TLS)
            node_id: Unique identifier for this client
            username: MQTT username (optional)
            password: MQTT password (optional)
            use_tls: Enable TLS encryption
            ca_cert: CA certificate path
            client_cert: Client certificate path
            client_key: Client key path
        """
        self.broker = broker
        self.port = port
        self.node_id = node_id
        self.username = username
        self.password = password
        self.use_tls = use_tls
        self.ca_cert = ca_cert
        self.client_cert = client_cert
        self.client_key = client_key

        self.client = None
        self.is_connected = False
        self._callbacks: Dict[str, Callable] = {}
        self._reconnect_thread = None

        self.logger = logging.getLogger(__name__)

        # Mock mode for development
        self._mock_mode = not MQTT_AVAILABLE

        if self._mock_mode:
            self.logger.warning(
                "paho-mqtt not available - running in mock mode. "
                "Install with: pip install paho-mqtt"
            )

    def connect(self, timeout: float = 10.0) -> bool:
        """
        Connect to MQTT broker.

        Args:
            timeout: Connection timeout in seconds

        Returns:
            True if connected successfully
        """
        if self._mock_mode:
            self.is_connected = True
            self.logger.info(f"Mock connected to {self.broker}:{self.port}")
            return True

        try:
            # Create client
            client_id = f"thesis_edge_{self.node_id}_{int(time.time())}"
            self.client = mqtt.Client(client_id=client_id)

            # Set callbacks
            self.client.on_connect = self._on_connect
            self.client.on_disconnect = self._on_disconnect
            self.client.on_message = self._on_message

            # Authentication
            if self.username and self.password:
                self.client.username_pw_set(self.username, self.password)

            # TLS
            if self.use_tls:
                self._setup_tls()

            # Connect
            self.client.connect(self.broker, self.port, keepalive=60)
            self.client.loop_start()

            # Wait for connection
            start = time.time()
            while not self.is_connected and time.time() - start < timeout:
                time.sleep(0.1)

            if self.is_connected:
                self.logger.info(
                    f"Connected to MQTT broker at {self.broker}:{self.port}"
                )
                return True
            else:
                self.logger.error("Connection timeout")
                return False

        except Exception as e:
            self.logger.error(f"Connection failed: {e}")
            return False

    def _setup_tls(self) -> None:
        """Setup TLS encryption."""
        context = ssl.create_default_context()

        if self.ca_cert:
            context.load_verify_locations(self.ca_cert)

        if self.client_cert and self.client_key:
            context.load_cert_chain(self.client_cert, self.client_key)

        self.client.tls_set_context(context)

    def _on_connect(self, client, userdata, flags, rc):
        """Callback when connected."""
        if rc == 0:
            self.is_connected = True
            self.logger.info("MQTT connected")

            # Resubscribe to topics
            for topic in self._callbacks:
                self.client.subscribe(topic)
        else:
            self.logger.error(f"Connection failed with code {rc}")

    def _on_disconnect(self, client, userdata, rc):
        """Callback when disconnected."""
        self.is_connected = False
        self.logger.warning(f"MQTT disconnected (rc={rc})")

        if rc != 0:
            # Unexpected disconnect - try to reconnect
            self._start_reconnect()

    def _on_message(self, client, userdata, message):
        """Callback when message received."""
        topic = message.topic
        payload = message.payload

        self.logger.debug(
            f"Received message on {topic}: {len(payload)} bytes"
        )

        # Find matching callback. Support MQTT wildcards (+ and #) by
        # delegating to paho's topic_matches_sub. A concrete topic like
        # "thesis/edge/A/data" must match a subscription like
        # "thesis/edge/+/data".
        for sub_topic, callback in self._callbacks.items():
            if sub_topic == topic or mqtt.topic_matches_sub(sub_topic, topic):
                try:
                    callback(topic, payload)
                except Exception as e:
                    self.logger.error(f"Callback error: {e}")
                return

    def _start_reconnect(self) -> None:
        """Start reconnection thread."""
        if self._reconnect_thread and self._reconnect_thread.is_alive():
            return

        def reconnect_loop():
            while not self.is_connected:
                try:
                    self.client.reconnect()
                    break
                except Exception as e:
                    self.logger.warning(f"Reconnect failed: {e}")
                    time.sleep(5)

        self._reconnect_thread = Thread(target=reconnect_loop, daemon=True)
        self._reconnect_thread.start()

    def disconnect(self) -> None:
        """Disconnect from broker."""
        if self._mock_mode:
            self.is_connected = False
            return

        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
            self.is_connected = False
            self.logger.info("Disconnected from MQTT broker")

    def publish(
        self,
        topic: str,
        payload: Any,
        qos: int = 1,
        retain: bool = False
    ) -> bool:
        """
        Publish message to topic.

        Args:
            topic: MQTT topic
            payload: Message payload (bytes, str, or dict)
            qos: Quality of service (0, 1, or 2)
            retain: Retain message on broker

        Returns:
            True if published successfully
        """
        if self._mock_mode:
            self.logger.info(
                f"Mock publish to {topic}: {len(str(payload))} bytes"
            )
            return True

        if not self.is_connected:
            self.logger.warning("Not connected - cannot publish")
            return False

        try:
            # Convert payload if needed
            if isinstance(payload, dict):
                payload = json.dumps(payload).encode('utf-8')
            elif isinstance(payload, str):
                payload = payload.encode('utf-8')

            result = self.client.publish(topic, payload, qos=qos, retain=retain)

            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                self.logger.debug(f"Published to {topic}")
                return True
            else:
                self.logger.error(f"Publish failed: {result.rc}")
                return False

        except Exception as e:
            self.logger.error(f"Publish error: {e}")
            return False

    def subscribe(
        self,
        topic: str,
        callback: Callable[[str, bytes], None],
        qos: int = 1
    ) -> bool:
        """
        Subscribe to topic with callback.

        Args:
            topic: MQTT topic (supports wildcards)
            callback: Function(topic, payload) to call on message
            qos: Quality of service

        Returns:
            True if subscribed successfully
        """
        self._callbacks[topic] = callback

        if self._mock_mode:
            self.logger.info(f"Mock subscribed to {topic}")
            return True

        if not self.is_connected:
            self.logger.warning("Not connected - will subscribe when connected")
            return False

        try:
            result = self.client.subscribe(topic, qos=qos)

            if result[0] == mqtt.MQTT_ERR_SUCCESS:
                self.logger.info(f"Subscribed to {topic}")
                return True
            else:
                self.logger.error(f"Subscribe failed: {result[0]}")
                return False

        except Exception as e:
            self.logger.error(f"Subscribe error: {e}")
            return False

    def unsubscribe(self, topic: str) -> bool:
        """Unsubscribe from topic."""
        if topic in self._callbacks:
            del self._callbacks[topic]

        if self._mock_mode:
            return True

        if self.client:
            self.client.unsubscribe(topic)

        return True

    def get_status(self) -> Dict[str, Any]:
        """Get client status."""
        return {
            'broker': self.broker,
            'port': self.port,
            'node_id': self.node_id,
            'connected': self.is_connected,
            'tls_enabled': self.use_tls,
            'subscriptions': list(self._callbacks.keys())
        }
