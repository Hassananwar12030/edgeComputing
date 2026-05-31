"""
Strategy Base Interface
=======================

Defines the abstract interface that all learning strategies must implement.

The strategy pattern allows switching between different learning approaches
(Centralized, Hybrid, Federated) without changing the main edge node logic.

Each strategy defines:
1. What to do when a sample is captured (on_sample)
2. What to do when the buffer is full (execute)
3. How to handle model updates from server (on_model_update)

Usage:
    class MyStrategy(ILearningStrategy):
        def execute(self, samples):
            # Do something with samples
            pass

    strategy = MyStrategy(mqtt_client, config)
    strategy.on_sample(sample)  # Called for each sample
    if strategy.should_trigger():
        strategy.execute(samples)  # Called when buffer full
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging


class ILearningStrategy(ABC):
    """
    Abstract interface for learning strategies.

    All strategies (Centralized, Hybrid, Federated) must implement
    this interface to ensure consistent behavior.

    The lifecycle is:
    1. on_sample() - Called for each captured sample
    2. should_trigger() - Check if action should be taken
    3. execute() - Perform the strategy-specific action
    4. on_model_update() - Handle new model from server
    """

    def __init__(
        self,
        node_id: str,
        mqtt_client=None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the strategy.

        Args:
            node_id: Unique identifier for this edge node
            mqtt_client: MQTT client for server communication
            config: Strategy-specific configuration
        """
        self.node_id = node_id
        self.mqtt_client = mqtt_client
        self.config = config or {}

        # Sample buffer
        self._buffer: List[Dict] = []
        self._buffer_size = self.config.get('buffer_size', 500)

        self.logger = logging.getLogger(self.__class__.__name__)

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the strategy name (e.g., 'centralized', 'hybrid', 'federated')."""
        pass

    @abstractmethod
    def prepare_sample(self, sample: Dict) -> Dict:
        """
        Prepare a sample for buffering/upload.

        Different strategies prepare samples differently:
        - Centralized: Keep raw audio
        - Hybrid: Convert to spectrogram
        - Federated: Convert to spectrogram (same as hybrid)

        Args:
            sample: Raw sample containing 'audio', 'label', 'timestamp', etc.

        Returns:
            Prepared sample dict
        """
        pass

    @abstractmethod
    def execute(self, samples: List[Dict]) -> bool:
        """
        Execute the strategy action.

        Called when buffer is full. Different for each strategy:
        - Centralized: Upload raw audio to server
        - Hybrid: Upload spectrograms to server
        - Federated: Train locally, upload weights

        Args:
            samples: List of prepared samples

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    def on_model_update(self, model_data: bytes) -> bool:
        """
        Handle model update from server.

        Called when server sends a new model. Different for each strategy:
        - Centralized: Replace local model
        - Hybrid: Replace local model
        - Federated: Replace local model (after aggregation)

        Args:
            model_data: Serialized model weights

        Returns:
            True if update successful, False otherwise
        """
        pass

    def on_sample(self, sample: Dict) -> None:
        """
        Handle a new captured sample.

        Prepares the sample and adds it to the buffer.

        Args:
            sample: Raw sample from capture pipeline
        """
        prepared = self.prepare_sample(sample)
        self._buffer.append(prepared)

        self.logger.debug(
            f"Sample buffered ({len(self._buffer)}/{self._buffer_size})"
        )

    def should_trigger(self) -> bool:
        """
        Check if the strategy action should be triggered.

        Default: Trigger when buffer reaches buffer_size.

        Returns:
            True if action should be triggered
        """
        return len(self._buffer) >= self._buffer_size

    def trigger(self) -> bool:
        """
        Trigger the strategy action if conditions are met.

        Convenience method that checks should_trigger() and
        calls execute() if needed.

        Returns:
            True if action was triggered and successful
        """
        if not self.should_trigger():
            return False

        # Get samples and clear buffer
        samples = self._buffer.copy()
        self._buffer.clear()

        # Execute strategy
        success = self.execute(samples)

        if not success:
            # Put samples back if failed
            self._buffer.extend(samples)

        return success

    def get_buffer_status(self) -> Dict[str, int]:
        """Get buffer status for monitoring."""
        return {
            'current_size': len(self._buffer),
            'max_size': self._buffer_size,
            'percentage': int(len(self._buffer) / self._buffer_size * 100)
        }

    def clear_buffer(self) -> int:
        """Clear the buffer and return number of discarded samples."""
        count = len(self._buffer)
        self._buffer.clear()
        return count

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get strategy-specific metrics.

        Override in subclasses to provide metrics like:
        - Bytes uploaded
        - Training time
        - Model accuracy
        """
        return {
            'strategy': self.name,
            'buffer_size': len(self._buffer)
        }
