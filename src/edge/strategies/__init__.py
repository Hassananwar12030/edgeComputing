"""
Learning Strategies Module
==========================

This module implements the three learning strategies compared in the thesis:

Strategy A (Centralized):
    - Edge: Capture + YOLOv8 labeling only
    - Upload: Raw audio + labels to server
    - Server: Full training (STFT + model training)
    - Privacy: LOW (raw audio leaves device)
    - Bandwidth: HIGH (~4MB per batch)

Strategy B (Hybrid STFT):
    - Edge: Capture + YOLOv8 + STFT processing
    - Upload: Spectrograms + labels to server
    - Server: Model training only
    - Privacy: MEDIUM (processed features leave device)
    - Bandwidth: MEDIUM (~3MB per batch)

Strategy C (Federated):
    - Edge: Capture + YOLOv8 + STFT + Local training
    - Upload: Model weights only to server
    - Server: FedAvg aggregation only
    - Privacy: HIGH (data never leaves device)
    - Bandwidth: LOW (~2MB per round)

All strategies implement the ILearningStrategy interface.

Classes:
    ILearningStrategy: Abstract interface for strategies
    CentralizedStrategy: Strategy A implementation
    HybridStrategy: Strategy B implementation
    FederatedStrategy: Strategy C implementation
    StrategyFactory: Factory to create strategies
"""

from .base import ILearningStrategy
from .centralized import CentralizedStrategy
from .hybrid import HybridStrategy
from .federated import FederatedStrategy

__all__ = [
    "ILearningStrategy",
    "CentralizedStrategy",
    "HybridStrategy",
    "FederatedStrategy"
]


class StrategyFactory:
    """Factory class to create strategy instances."""

    _strategies = {
        "centralized": CentralizedStrategy,
        "hybrid": HybridStrategy,
        "federated": FederatedStrategy,
        "a": CentralizedStrategy,
        "b": HybridStrategy,
        "c": FederatedStrategy
    }

    @classmethod
    def create(cls, strategy_type: str, **kwargs):
        """
        Create a strategy instance.

        Args:
            strategy_type: One of "centralized", "hybrid", "federated"
                          or "a", "b", "c"
            **kwargs: Arguments passed to strategy constructor

        Returns:
            Strategy instance

        Raises:
            ValueError: If strategy_type is unknown
        """
        strategy_type = strategy_type.lower()
        if strategy_type not in cls._strategies:
            valid = list(cls._strategies.keys())
            raise ValueError(
                f"Unknown strategy: {strategy_type}. Valid: {valid}"
            )

        return cls._strategies[strategy_type](**kwargs)
