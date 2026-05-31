"""
Aggregation Module
==================

Implements federated learning aggregation algorithms.

Primary algorithm: FedAvg (Federated Averaging)
    - Weighted average of client model weights
    - Weights proportional to number of training samples

Alternative algorithms (for future):
    - FedProx: Adds proximal term for heterogeneous data
    - FedNova: Normalizes for different local epochs
    - Scaffold: Variance reduction technique

Usage:
    aggregator = FedAvgAggregator()
    global_weights = aggregator.aggregate(client_weights, sample_counts)
"""

from .fedavg import FedAvgAggregator

__all__ = ["FedAvgAggregator"]
