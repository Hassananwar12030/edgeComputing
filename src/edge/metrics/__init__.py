"""
Metrics Module
==============

Handles measurement and reporting of performance metrics on edge devices.

Key metrics collected:
1. Power consumption (via INA219 sensor)
2. Network bandwidth usage
3. Inference/training latency
4. Memory usage
5. CPU/temperature monitoring

These metrics are crucial for comparing strategies:
- Strategy A: High bandwidth, low edge compute
- Strategy B: Medium bandwidth, medium edge compute
- Strategy C: Low bandwidth, high edge compute

Classes:
    PowerMonitor: Power consumption via INA219
    BandwidthTracker: Network usage tracking
    LatencyMeasure: Timing measurements
    MetricsCollector: Aggregates all metrics
"""

from .power import PowerMonitor
from .bandwidth import BandwidthTracker
from .latency import LatencyMeasure
from .collector import MetricsCollector

__all__ = [
    "PowerMonitor",
    "BandwidthTracker",
    "LatencyMeasure",
    "MetricsCollector"
]
