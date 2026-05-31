"""
Server Main Entry Point
========================

Main orchestrator for the central server. Supports:
- Strategy A/B: Centralized training with MQTT data reception
- Strategy C: Flower server for federated learning

The server:
1. Receives data/weights from edge nodes
2. Performs training or aggregation
3. Broadcasts updated model

Usage:
    # Centralized mode (Strategy A/B)
    python -m src.server.main --strategy centralized --port 1883

    # Federated mode (Strategy C)
    python -m src.server.main --strategy federated --port 8080

Author: Adnan Anwar Rajput
"""

import argparse
import logging
import signal
import sys
from typing import Optional, Dict, Any

# These imports will be implemented in subsequent files
# from .flower_server import FlowerServer
# from .training import CentralizedTrainer
# from .api import MQTTHandler


class Server:
    """
    Central server for federated learning project.

    Supports multiple modes:
    - Centralized: Receives data, trains locally
    - Federated: Coordinates FL with Flower

    Attributes:
        mode: "centralized" or "federated"
        host: Server host address
        port: Server port
    """

    def __init__(
        self,
        mode: str = "federated",
        host: str = "0.0.0.0",
        port: int = 8080,
        config_path: Optional[str] = None
    ):
        """
        Initialize the server.

        Args:
            mode: Operating mode (centralized/federated)
            host: Host address to bind to
            port: Port to listen on
            config_path: Path to configuration file
        """
        self.mode = mode
        self.host = host
        self.port = port
        self.config_path = config_path

        # Components (initialized in setup)
        self.flower_server = None
        self.trainer = None
        self.mqtt_handler = None

        self.is_running = False

        # Setup logging
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Configure logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] [SERVER] %(levelname)s: %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('logs/server.log')
            ]
        )
        self.logger = logging.getLogger(__name__)

    def setup(self) -> None:
        """
        Initialize server components based on mode.
        """
        self.logger.info(f"Setting up server in {self.mode} mode...")

        if self.mode == "federated":
            # Setup Flower server
            # self.flower_server = FlowerServer(
            #     host=self.host,
            #     port=self.port
            # )
            self.logger.info("Flower server initialized")

        elif self.mode == "centralized":
            # Setup centralized training components
            # self.trainer = CentralizedTrainer()
            # self.mqtt_handler = MQTTHandler(self.trainer)
            self.logger.info("Centralized trainer initialized")

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        self.logger.info("Server setup complete")

    def run(self) -> None:
        """
        Start the server.

        For federated mode: Starts Flower server
        For centralized mode: Starts MQTT listener and training loop
        """
        self.is_running = True
        self.logger.info(f"Starting server on {self.host}:{self.port}...")

        try:
            if self.mode == "federated":
                self._run_federated()
            else:
                self._run_centralized()

        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal")
        finally:
            self.shutdown()

    def _run_federated(self) -> None:
        """Run Flower federated learning server."""
        self.logger.info("Starting Flower server...")

        # TODO: Start Flower server
        # self.flower_server.start()

        # Placeholder
        import time
        while self.is_running:
            time.sleep(1)

    def _run_centralized(self) -> None:
        """Run centralized training server."""
        self.logger.info("Starting centralized training server...")

        # TODO: Start MQTT handler and training loop
        # self.mqtt_handler.start()
        # self.trainer.training_loop()

        # Placeholder
        import time
        while self.is_running:
            time.sleep(1)

    def shutdown(self) -> None:
        """Graceful shutdown."""
        self.logger.info("Shutting down server...")
        self.is_running = False

        # Cleanup components
        # if self.flower_server:
        #     self.flower_server.stop()
        # if self.mqtt_handler:
        #     self.mqtt_handler.stop()

        self.logger.info("Server shutdown complete")

    def get_status(self) -> Dict[str, Any]:
        """Get server status."""
        return {
            'mode': self.mode,
            'host': self.host,
            'port': self.port,
            'is_running': self.is_running
        }


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Central Server for Federated Learning"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["centralized", "federated"],
        default="federated",
        help="Server operating mode"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address to bind to"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to listen on"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/server_config.yaml",
        help="Path to configuration file"
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Create and run server
    server = Server(
        mode=args.mode,
        host=args.host,
        port=args.port,
        config_path=args.config
    )

    # Setup signal handlers
    signal.signal(signal.SIGINT, lambda s, f: server.shutdown())
    signal.signal(signal.SIGTERM, lambda s, f: server.shutdown())

    # Initialize and run
    server.setup()
    server.run()


if __name__ == "__main__":
    main()
