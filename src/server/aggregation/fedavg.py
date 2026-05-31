"""
FedAvg Aggregation
==================

Implements the Federated Averaging (FedAvg) algorithm.

FedAvg paper: "Communication-Efficient Learning of Deep Networks
              from Decentralized Data" (McMahan et al., 2017)

Algorithm:
    1. Server sends global model to selected clients
    2. Each client trains locally for E epochs on B batches
    3. Clients send updated weights back to server
    4. Server aggregates: w_global = Σ (n_k / n) * w_k
       where n_k = samples on client k, n = total samples

Properties:
    - Communication efficient (only weights, not data)
    - Privacy preserving (data stays on device)
    - Works with non-IID data (with some degradation)
"""

import logging
from typing import List, Tuple, Dict, Any, Optional
import numpy as np


class FedAvgAggregator:
    """
    FedAvg aggregation implementation.

    Performs weighted averaging of model weights based on
    the number of training samples each client used.

    Attributes:
        global_weights: Current global model weights
        round_count: Number of aggregation rounds
    """

    def __init__(
        self,
        initial_weights: Optional[List[np.ndarray]] = None
    ):
        """
        Initialize FedAvg aggregator.

        Args:
            initial_weights: Initial global model weights
        """
        self.global_weights = initial_weights
        self.round_count = 0
        self.logger = logging.getLogger(__name__)

        # Track history
        self.history: List[Dict[str, Any]] = []

    def aggregate(
        self,
        client_weights: List[List[np.ndarray]],
        sample_counts: List[int],
        client_metrics: Optional[List[Dict[str, float]]] = None
    ) -> List[np.ndarray]:
        """
        Aggregate client weights using FedAvg.

        Args:
            client_weights: List of weight arrays from each client
            sample_counts: Number of samples each client trained on
            client_metrics: Optional metrics from each client

        Returns:
            Aggregated global weights
        """
        if not client_weights:
            self.logger.warning("No client weights to aggregate")
            return self.global_weights or []

        n_clients = len(client_weights)
        total_samples = sum(sample_counts)

        if total_samples == 0:
            self.logger.warning("Total samples is 0, using equal weights")
            sample_counts = [1] * n_clients
            total_samples = n_clients

        self.logger.info(
            f"Aggregating weights from {n_clients} clients "
            f"({total_samples} total samples)"
        )

        # Calculate weighted average
        aggregated = []
        n_layers = len(client_weights[0])

        for layer_idx in range(n_layers):
            # Get this layer's weights from all clients
            layer_weights = [
                client_weights[i][layer_idx]
                for i in range(n_clients)
            ]

            # Weighted average
            weighted_sum = np.zeros_like(layer_weights[0])
            for i, weights in enumerate(layer_weights):
                weight_factor = sample_counts[i] / total_samples
                weighted_sum += weight_factor * weights

            aggregated.append(weighted_sum)

        self.global_weights = aggregated
        self.round_count += 1

        # Record history
        round_info = {
            'round': self.round_count,
            'n_clients': n_clients,
            'total_samples': total_samples,
            'sample_distribution': sample_counts
        }

        if client_metrics:
            round_info['avg_accuracy'] = np.mean(
                [m.get('accuracy', 0) for m in client_metrics]
            )
            round_info['avg_loss'] = np.mean(
                [m.get('loss', 0) for m in client_metrics]
            )

        self.history.append(round_info)
        self.logger.info(f"Aggregation round {self.round_count} complete")

        return aggregated

    def aggregate_fit_results(
        self,
        results: List[Tuple[List[np.ndarray], int, Dict[str, float]]]
    ) -> Tuple[List[np.ndarray], Dict[str, float]]:
        """
        Aggregate results in Flower format.

        Args:
            results: List of (weights, num_examples, metrics) tuples

        Returns:
            Tuple of (aggregated_weights, aggregated_metrics)
        """
        if not results:
            return self.global_weights or [], {}

        # Extract components
        client_weights = [r[0] for r in results]
        sample_counts = [r[1] for r in results]
        client_metrics = [r[2] for r in results]

        # Aggregate weights
        aggregated_weights = self.aggregate(
            client_weights, sample_counts, client_metrics
        )

        # Aggregate metrics (weighted average)
        total_samples = sum(sample_counts)
        aggregated_metrics = {}

        metric_keys = set()
        for metrics in client_metrics:
            metric_keys.update(metrics.keys())

        for key in metric_keys:
            weighted_sum = sum(
                sample_counts[i] * client_metrics[i].get(key, 0)
                for i in range(len(results))
            )
            aggregated_metrics[key] = weighted_sum / total_samples

        return aggregated_weights, aggregated_metrics

    def get_global_weights(self) -> Optional[List[np.ndarray]]:
        """Get current global weights."""
        return self.global_weights

    def set_global_weights(self, weights: List[np.ndarray]) -> None:
        """Set global weights."""
        self.global_weights = weights

    def get_history(self) -> List[Dict[str, Any]]:
        """Get aggregation history."""
        return self.history.copy()

    def reset(self) -> None:
        """Reset aggregator state."""
        self.global_weights = None
        self.round_count = 0
        self.history.clear()


class WeightedFedAvg(FedAvgAggregator):
    """
    Weighted FedAvg with custom weighting schemes.

    Supports different weighting strategies:
    - 'samples': Weight by number of samples (default FedAvg)
    - 'equal': Equal weight to all clients
    - 'performance': Weight by client performance (accuracy)
    """

    def __init__(
        self,
        weighting_scheme: str = 'samples',
        initial_weights: Optional[List[np.ndarray]] = None
    ):
        """
        Initialize weighted FedAvg.

        Args:
            weighting_scheme: One of 'samples', 'equal', 'performance'
            initial_weights: Initial global weights
        """
        super().__init__(initial_weights)
        self.weighting_scheme = weighting_scheme

    def _calculate_weights(
        self,
        sample_counts: List[int],
        client_metrics: Optional[List[Dict[str, float]]] = None
    ) -> List[float]:
        """Calculate client weights based on scheme."""
        n_clients = len(sample_counts)

        if self.weighting_scheme == 'equal':
            return [1.0 / n_clients] * n_clients

        elif self.weighting_scheme == 'performance':
            if client_metrics is None:
                # Fall back to sample weighting
                total = sum(sample_counts)
                return [s / total for s in sample_counts]

            # Weight by accuracy (higher accuracy = more weight)
            accuracies = [
                max(0.1, m.get('accuracy', 0.5))
                for m in client_metrics
            ]
            total = sum(accuracies)
            return [a / total for a in accuracies]

        else:  # 'samples' - default
            total = sum(sample_counts)
            if total == 0:
                return [1.0 / n_clients] * n_clients
            return [s / total for s in sample_counts]
