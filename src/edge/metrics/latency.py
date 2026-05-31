"""
Latency Measurement Module
==========================

Measures timing for various operations.

Key latencies tracked:
1. Inference latency (YOLOv8 detection time)
2. STFT processing time
3. Training iteration time
4. Network round-trip time
5. End-to-end pipeline latency

Usage:
    measure = LatencyMeasure()

    with measure.time("inference"):
        model.predict(frame)

    print(measure.get_average("inference"))
"""

import logging
import time
import threading
from typing import Dict, Any, Optional, List
from collections import deque, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class LatencyRecord:
    """Record of a timed operation."""
    operation: str
    duration_ms: float
    timestamp: float
    metadata: Optional[Dict[str, Any]] = None


class LatencyMeasure:
    """
    Latency measurement and tracking.

    Provides context managers and decorators for timing operations,
    with automatic statistics calculation.

    Attributes:
        records: Recent latency records by operation
        active_timers: Currently running timers
    """

    def __init__(self, buffer_size: int = 1000):
        """
        Initialize latency measure.

        Args:
            buffer_size: Number of records to keep per operation
        """
        self.buffer_size = buffer_size
        self.records: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=buffer_size)
        )
        self._active_timers: Dict[str, float] = {}
        self._lock = threading.Lock()

        self.logger = logging.getLogger(__name__)

    @contextmanager
    def time(
        self,
        operation: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Context manager for timing an operation.

        Args:
            operation: Name of the operation
            metadata: Additional metadata to record

        Usage:
            with measure.time("inference"):
                result = model.predict(x)
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = (time.perf_counter() - start) * 1000  # ms
            self._record(operation, duration, metadata)

    def start(self, operation: str) -> None:
        """
        Start a timer for an operation.

        Use with stop() for timing across non-contiguous code.

        Args:
            operation: Name of the operation
        """
        with self._lock:
            self._active_timers[operation] = time.perf_counter()

    def stop(
        self,
        operation: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Stop a timer and record the duration.

        Args:
            operation: Name of the operation
            metadata: Additional metadata

        Returns:
            Duration in milliseconds
        """
        with self._lock:
            if operation not in self._active_timers:
                self.logger.warning(f"Timer not started: {operation}")
                return 0.0

            start = self._active_timers.pop(operation)
            duration = (time.perf_counter() - start) * 1000

        self._record(operation, duration, metadata)
        return duration

    def _record(
        self,
        operation: str,
        duration_ms: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a latency measurement."""
        record = LatencyRecord(
            operation=operation,
            duration_ms=duration_ms,
            timestamp=time.time(),
            metadata=metadata
        )

        with self._lock:
            self.records[operation].append(record)

        self.logger.debug(f"{operation}: {duration_ms:.2f}ms")

    def record(
        self,
        operation: str,
        duration_ms: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Manually record a latency.

        Args:
            operation: Operation name
            duration_ms: Duration in milliseconds
            metadata: Additional metadata
        """
        self._record(operation, duration_ms, metadata)

    def get_last(self, operation: str) -> Optional[float]:
        """Get most recent latency for operation."""
        with self._lock:
            if operation in self.records and self.records[operation]:
                return self.records[operation][-1].duration_ms
        return None

    def get_average(
        self,
        operation: str,
        last_n: Optional[int] = None
    ) -> float:
        """
        Get average latency for operation.

        Args:
            operation: Operation name
            last_n: Only consider last N samples (None = all)

        Returns:
            Average latency in milliseconds
        """
        with self._lock:
            if operation not in self.records:
                return 0.0

            records = list(self.records[operation])

        if not records:
            return 0.0

        if last_n:
            records = records[-last_n:]

        return sum(r.duration_ms for r in records) / len(records)

    def get_percentile(
        self,
        operation: str,
        percentile: float = 95.0
    ) -> float:
        """
        Get percentile latency.

        Args:
            operation: Operation name
            percentile: Percentile (0-100)

        Returns:
            Latency at given percentile in milliseconds
        """
        with self._lock:
            if operation not in self.records:
                return 0.0

            durations = sorted(r.duration_ms for r in self.records[operation])

        if not durations:
            return 0.0

        idx = int(len(durations) * percentile / 100)
        idx = min(idx, len(durations) - 1)
        return durations[idx]

    def get_stats(self, operation: str) -> Dict[str, float]:
        """
        Get comprehensive statistics for operation.

        Returns dict with: avg, min, max, p50, p95, p99, count
        """
        with self._lock:
            if operation not in self.records or not self.records[operation]:
                return {}

            durations = [r.duration_ms for r in self.records[operation]]

        durations_sorted = sorted(durations)
        n = len(durations_sorted)

        return {
            'avg_ms': sum(durations) / n,
            'min_ms': min(durations),
            'max_ms': max(durations),
            'p50_ms': durations_sorted[n // 2],
            'p95_ms': durations_sorted[int(n * 0.95)],
            'p99_ms': durations_sorted[int(n * 0.99)],
            'count': n
        }

    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all operations."""
        with self._lock:
            operations = list(self.records.keys())

        return {op: self.get_stats(op) for op in operations}

    def get_operations(self) -> List[str]:
        """Get list of tracked operations."""
        with self._lock:
            return list(self.records.keys())

    def clear(self, operation: Optional[str] = None) -> None:
        """
        Clear latency records.

        Args:
            operation: Specific operation (None = all)
        """
        with self._lock:
            if operation:
                if operation in self.records:
                    self.records[operation].clear()
            else:
                self.records.clear()

    def timed(self, operation: str):
        """
        Decorator for timing a function.

        Args:
            operation: Name for this operation

        Usage:
            @measure.timed("my_function")
            def my_function():
                pass
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                with self.time(operation):
                    return func(*args, **kwargs)
            return wrapper
        return decorator
