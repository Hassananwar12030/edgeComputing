"""
Flower Server Module
====================

Implements the Flower federated learning server.

This server coordinates federated learning for Strategy C:
1. Waits for clients to connect
2. Sends global model to clients
3. Receives updated weights from clients
4. Aggregates using FedAvg
5. Repeats for N rounds

Configuration:
    - num_rounds: Number of FL rounds (default: 10)
    - min_clients: Minimum clients per round (default: 2)
    - fraction_fit: Fraction of clients for training (default: 1.0)

Usage:
    server = FlowerServer(host="0.0.0.0", port=8080)
    server.start()

    # Or from command line:
    python -m src.server.flower_server --rounds 10 --min-clients 2
"""

import logging
import argparse
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

# Flower framework
try:
    import flwr as fl
    from flwr.common import (
        Parameters,
        Metrics,
        FitRes,
        EvaluateRes,
        ndarrays_to_parameters,
        parameters_to_ndarrays,
    )
    from flwr.server.strategy import FedAvg
    from flwr.server import ServerConfig
    FLOWER_AVAILABLE = True
except ImportError:
    FLOWER_AVAILABLE = False
    fl = None


class FlowerServer:
    """
    Flower federated learning server.

    Coordinates federated learning rounds with edge clients.

    Attributes:
        host: Server host address
        port: Server port
        num_rounds: Number of FL rounds
        min_clients: Minimum clients per round
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8080,
        num_rounds: int = 10,
        min_clients: int = 2,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        initial_model_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Flower server.

        Args:
            host: Host address to bind
            port: Port to listen on
            num_rounds: Number of FL rounds
            min_clients: Minimum clients required
            fraction_fit: Fraction of clients for training
            fraction_evaluate: Fraction of clients for evaluation
            initial_model_path: Path to initial model weights
            config: Additional configuration
        """
        self.host = host
        self.port = port
        self.num_rounds = num_rounds
        self.min_clients = min_clients
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.initial_model_path = initial_model_path
        self.config = config or {}

        self.logger = logging.getLogger(__name__)

        # Track metrics across rounds
        self.round_metrics: List[Dict] = []

        if not FLOWER_AVAILABLE:
            self.logger.error(
                "Flower not available. Install with: pip install flwr"
            )

    def start(self) -> None:
        """
        Start the Flower server.

        This is a blocking call that runs until all rounds complete.
        """
        if not FLOWER_AVAILABLE:
            self.logger.error("Cannot start - Flower not installed")
            return

        self.logger.info(
            f"Starting Flower server on {self.host}:{self.port}"
        )
        self.logger.info(
            f"Config: {self.num_rounds} rounds, "
            f"min {self.min_clients} clients"
        )

        # Create strategy
        strategy = self._create_strategy()

        # Server configuration
        server_config = ServerConfig(num_rounds=self.num_rounds)

        # Start server
        fl.server.start_server(
            server_address=f"{self.host}:{self.port}",
            config=server_config,
            strategy=strategy
        )

        self.logger.info("Flower server completed all rounds")

    def _create_strategy(self):
        """
        Create the Flower strategy (FedAvg).

        Returns:
            Configured FedAvg strategy
        """
        # Load initial parameters if available
        initial_parameters = None
        if self.initial_model_path:
            try:
                initial_parameters = self._load_initial_parameters()
            except Exception as e:
                self.logger.warning(f"Could not load initial model: {e}")

        # Create custom FedAvg strategy with callbacks
        strategy = FedAvg(
            fraction_fit=self.fraction_fit,
            fraction_evaluate=self.fraction_evaluate,
            min_fit_clients=self.min_clients,
            min_evaluate_clients=self.min_clients,
            min_available_clients=self.min_clients,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=self._aggregate_fit_metrics,
            evaluate_metrics_aggregation_fn=self._aggregate_evaluate_metrics,
            on_fit_config_fn=self._get_fit_config,
            on_evaluate_config_fn=self._get_evaluate_config,
        )

        return strategy

    def _load_initial_parameters(self):
        """Load initial model parameters."""
        # TODO: Load model and extract weights
        # model = tf.keras.models.load_model(self.initial_model_path)
        # weights = model.get_weights()
        # return ndarrays_to_parameters(weights)
        return None

    def _get_fit_config(self, server_round: int) -> Dict[str, Any]:
        """
        Get configuration for fit round.

        Args:
            server_round: Current round number

        Returns:
            Configuration dict for clients
        """
        config = {
            "round": server_round,
            "local_epochs": self.config.get("local_epochs", 3),
            "batch_size": self.config.get("batch_size", 8),
            "learning_rate": self.config.get("learning_rate", 0.001)
        }

        self.logger.info(f"Round {server_round} fit config: {config}")
        return config

    def _get_evaluate_config(self, server_round: int) -> Dict[str, Any]:
        """Get configuration for evaluate round."""
        return {"round": server_round}

    def _aggregate_fit_metrics(
        self,
        results: List[Tuple[int, Metrics]]
    ) -> Metrics:
        """
        Aggregate training metrics from all clients.

        Args:
            results: List of (num_examples, metrics) per client

        Returns:
            Aggregated metrics
        """
        if not results:
            return {}

        # Weighted average of metrics
        total_examples = sum(n for n, _ in results)
        aggregated = {}

        # Get all metric keys
        metric_keys = set()
        for _, metrics in results:
            metric_keys.update(metrics.keys())

        # Aggregate each metric
        for key in metric_keys:
            weighted_sum = 0.0
            for num_examples, metrics in results:
                if key in metrics:
                    weighted_sum += num_examples * metrics[key]
            aggregated[key] = weighted_sum / total_examples

        self.logger.info(f"Aggregated fit metrics: {aggregated}")
        self.round_metrics.append({
            'type': 'fit',
            'metrics': aggregated,
            'num_clients': len(results),
            'total_examples': total_examples
        })

        return aggregated

    def _aggregate_evaluate_metrics(
        self,
        results: List[Tuple[int, Metrics]]
    ) -> Metrics:
        """Aggregate evaluation metrics from all clients."""
        if not results:
            return {}

        total_examples = sum(n for n, _ in results)
        aggregated = {}

        metric_keys = set()
        for _, metrics in results:
            metric_keys.update(metrics.keys())

        for key in metric_keys:
            weighted_sum = 0.0
            for num_examples, metrics in results:
                if key in metrics:
                    weighted_sum += num_examples * metrics[key]
            aggregated[key] = weighted_sum / total_examples

        self.logger.info(f"Aggregated evaluate metrics: {aggregated}")
        return aggregated

    def get_round_history(self) -> List[Dict]:
        """Get metrics history for all rounds."""
        return self.round_metrics.copy()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Flower Federated Learning Server"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port number"
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=10,
        help="Number of FL rounds"
    )
    parser.add_argument(
        "--min-clients",
        type=int,
        default=2,
        help="Minimum clients per round"
    )
    parser.add_argument(
        "--local-epochs",
        type=int,
        default=3,
        help="Local training epochs"
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s'
    )

    args = parse_args()

    server = FlowerServer(
        host=args.host,
        port=args.port,
        num_rounds=args.rounds,
        min_clients=args.min_clients,
        config={"local_epochs": args.local_epochs}
    )

    server.start()


if __name__ == "__main__":
    main()
