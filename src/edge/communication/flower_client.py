"""
Flower Client Module
====================

Implements a Flower (flwr) client for federated learning.

Flower is the FL framework used for Strategy C:
- Handles client-server coordination
- Manages training rounds
- Provides weight serialization
- Supports various aggregation strategies

FL Workflow:
1. Server starts Flower server
2. Edge nodes connect as Flower clients
3. Server sends global model
4. Clients train locally
5. Clients send weights back
6. Server aggregates (FedAvg)
7. Repeat

Usage:
    client = FlowerClient(
        node_id="A",
        server_address="192.168.1.100:8080",
        trainer=local_trainer
    )
    client.start()  # Blocks until FL complete
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

# Flower framework
try:
    import flwr as fl
    from flwr.common import (
        Parameters,
        FitIns,
        FitRes,
        EvaluateIns,
        EvaluateRes,
        GetParametersIns,
        GetParametersRes,
        Status,
        Code,
        ndarrays_to_parameters,
        parameters_to_ndarrays,
    )
    FLOWER_AVAILABLE = True
except ImportError:
    FLOWER_AVAILABLE = False
    fl = None


class FlowerClient:
    """
    Flower federated learning client wrapper.

    Wraps the local trainer to work with Flower's FL protocol.

    Attributes:
        node_id: Edge node identifier
        server_address: Flower server address (host:port)
        trainer: LocalTrainer instance
    """

    def __init__(
        self,
        node_id: str,
        server_address: str = "localhost:8080",
        trainer=None,
        dataset=None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Flower client.

        Args:
            node_id: Edge node identifier
            server_address: Flower server address
            trainer: LocalTrainer instance
            dataset: LocalDataset instance
            config: Additional configuration
        """
        self.node_id = node_id
        self.server_address = server_address
        self.trainer = trainer
        self.dataset = dataset
        self.config = config or {}

        self.logger = logging.getLogger(__name__)

        # Mock mode for development
        self._mock_mode = not FLOWER_AVAILABLE

        if self._mock_mode:
            self.logger.warning(
                "Flower not available - running in mock mode. "
                "Install with: pip install flwr"
            )

    def start(self) -> None:
        """
        Start the Flower client.

        This connects to the Flower server and participates
        in federated learning rounds until the server stops.

        Note: This is a blocking call.
        """
        if self._mock_mode:
            self.logger.info("Mock Flower client started")
            return

        self.logger.info(
            f"Starting Flower client, connecting to {self.server_address}"
        )

        # Create Flower client
        client = FlowerClientImpl(
            node_id=self.node_id,
            trainer=self.trainer,
            dataset=self.dataset
        )

        # Start client (blocking)
        fl.client.start_numpy_client(
            server_address=self.server_address,
            client=client
        )

    def send_weights(
        self,
        weights: List[np.ndarray],
        metrics: Dict[str, float]
    ) -> bool:
        """
        Send weights to server (for non-Flower communication).

        This is a fallback method when not using Flower's protocol.

        Args:
            weights: Model weights
            metrics: Training metrics

        Returns:
            True if successful
        """
        if self._mock_mode:
            total_bytes = sum(w.nbytes for w in weights)
            self.logger.info(
                f"Mock: Sent {len(weights)} weight arrays "
                f"({total_bytes/1024:.1f} KB)"
            )
            return True

        # In normal operation, Flower handles this
        self.logger.warning("send_weights called outside Flower protocol")
        return False


# Actual Flower client implementation
if FLOWER_AVAILABLE:
    class FlowerClientImpl(fl.client.NumPyClient):
        """
        Flower NumPyClient implementation.

        This class implements Flower's client interface:
        - get_parameters: Return current model weights
        - fit: Train on local data, return updated weights
        - evaluate: Evaluate model on local data
        """

        def __init__(
            self,
            node_id: str,
            trainer,
            dataset
        ):
            """
            Initialize Flower client implementation.

            Args:
                node_id: Edge node identifier
                trainer: LocalTrainer instance
                dataset: LocalDataset instance
            """
            self.node_id = node_id
            self.trainer = trainer
            self.dataset = dataset
            self.logger = logging.getLogger(__name__)

        def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
            """
            Return current model parameters.

            Called by server at the start of a round.

            Args:
                config: Configuration from server

            Returns:
                List of model weight arrays
            """
            self.logger.info("Sending model parameters to server")
            return self.trainer.get_weights()

        def fit(
            self,
            parameters: List[np.ndarray],
            config: Dict[str, Any]
        ) -> Tuple[List[np.ndarray], int, Dict[str, float]]:
            """
            Train model on local data.

            Called by server for each training round.

            Args:
                parameters: Global model weights from server
                config: Training configuration

            Returns:
                Tuple of (updated_weights, num_samples, metrics)
            """
            self.logger.info("Received global model, starting local training")

            # Update model with global weights
            self.trainer.set_weights(parameters)

            # Get training data
            X, y = self.dataset.get_training_data()

            if len(X) == 0:
                self.logger.warning("No training data available")
                return self.trainer.get_weights(), 0, {}

            # Train locally
            epochs = config.get("local_epochs", 3)
            metrics = self.trainer.train(X, y, epochs=epochs)

            # Return updated weights
            updated_weights = self.trainer.get_weights()

            self.logger.info(
                f"Local training complete: {len(X)} samples, "
                f"accuracy={metrics.get('accuracy', 0):.3f}"
            )

            return updated_weights, len(X), metrics

        def evaluate(
            self,
            parameters: List[np.ndarray],
            config: Dict[str, Any]
        ) -> Tuple[float, int, Dict[str, float]]:
            """
            Evaluate model on local data.

            Called by server for evaluation rounds.

            Args:
                parameters: Model weights to evaluate
                config: Evaluation configuration

            Returns:
                Tuple of (loss, num_samples, metrics)
            """
            self.logger.info("Evaluating model on local data")

            # Set weights
            self.trainer.set_weights(parameters)

            # Get evaluation data
            X, y = self.dataset.get_training_data()

            if len(X) == 0:
                return 0.0, 0, {}

            # Evaluate
            loss, accuracy = self.trainer._evaluate(X, y)

            metrics = {
                "accuracy": accuracy,
                "node_id": self.node_id
            }

            return float(loss), len(X), metrics
else:
    # Stub for when Flower is not available
    class FlowerClientImpl:
        def __init__(self, *args, **kwargs):
            pass
