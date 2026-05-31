"""
Metrics Collector Module
========================

Aggregates all metrics from edge device for reporting and analysis.

Collects:
- Power consumption
- Bandwidth usage
- Latency measurements
- System stats (CPU, memory, temperature)

Outputs to:
- Console/logs
- MQTT (for central collection)
- CSV/JSON files (for offline analysis)

Usage:
    collector = MetricsCollector(node_id="A")
    collector.start()
    # ... edge node runs ...
    collector.report()
    collector.save("metrics.csv")
"""

import logging
import time
import json
import csv
import os
import threading
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime

# System monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


@dataclass
class MetricsSnapshot:
    """Snapshot of all metrics at a point in time."""
    timestamp: float
    node_id: str

    # Power
    power_mw: float
    energy_mwh: float

    # Bandwidth
    upload_bytes: int
    download_bytes: int
    upload_rate_bps: float

    # Latency
    inference_ms: float
    stft_ms: float
    training_ms: float

    # System
    cpu_percent: float
    memory_percent: float
    temperature_c: float

    # Strategy specific
    strategy: str
    samples_processed: int
    training_rounds: int


class MetricsCollector:
    """
    Central metrics collector for edge node.

    Aggregates metrics from power, bandwidth, and latency monitors
    and provides unified reporting interface.

    Attributes:
        node_id: Edge node identifier
        power_monitor: PowerMonitor instance
        bandwidth_tracker: BandwidthTracker instance
        latency_measure: LatencyMeasure instance
    """

    def __init__(
        self,
        node_id: str,
        power_monitor=None,
        bandwidth_tracker=None,
        latency_measure=None,
        mqtt_client=None,
        report_interval: float = 60.0,
        output_dir: str = "logs/metrics"
    ):
        """
        Initialize metrics collector.

        Args:
            node_id: Edge node identifier
            power_monitor: PowerMonitor instance
            bandwidth_tracker: BandwidthTracker instance
            latency_measure: LatencyMeasure instance
            mqtt_client: MQTT client for reporting
            report_interval: Seconds between auto-reports
            output_dir: Directory for output files
        """
        self.node_id = node_id
        self.power_monitor = power_monitor
        self.bandwidth_tracker = bandwidth_tracker
        self.latency_measure = latency_measure
        self.mqtt_client = mqtt_client
        self.report_interval = report_interval
        self.output_dir = output_dir

        self.snapshots: List[MetricsSnapshot] = []
        self._reporter_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Strategy tracking
        self.strategy = "unknown"
        self.samples_processed = 0
        self.training_rounds = 0

        self.logger = logging.getLogger(__name__)

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

    def start(self, auto_report: bool = True) -> None:
        """
        Start metrics collection.

        Args:
            auto_report: Enable automatic periodic reporting
        """
        # Start component monitors
        if self.power_monitor:
            self.power_monitor.start()

        # Start auto-reporter
        if auto_report:
            self._stop_event.clear()
            self._reporter_thread = threading.Thread(
                target=self._report_loop,
                daemon=True
            )
            self._reporter_thread.start()

        self.logger.info("Metrics collection started")

    def stop(self) -> None:
        """Stop metrics collection."""
        self._stop_event.set()

        if self._reporter_thread:
            self._reporter_thread.join(timeout=2.0)

        if self.power_monitor:
            self.power_monitor.stop()

        self.logger.info("Metrics collection stopped")

    def _report_loop(self) -> None:
        """Background reporting loop."""
        while not self._stop_event.is_set():
            try:
                self.report()
            except Exception as e:
                self.logger.error(f"Report error: {e}")

            # Wait for interval
            self._stop_event.wait(self.report_interval)

    def collect_snapshot(self) -> MetricsSnapshot:
        """
        Collect current metrics snapshot.

        Returns:
            MetricsSnapshot with all current values
        """
        # Power metrics
        power_mw = 0.0
        energy_mwh = 0.0
        if self.power_monitor:
            power_mw = self.power_monitor.get_current_power()
            energy_mwh = self.power_monitor.get_energy_consumed()

        # Bandwidth metrics
        upload_bytes = 0
        download_bytes = 0
        upload_rate = 0.0
        if self.bandwidth_tracker:
            upload_bytes = self.bandwidth_tracker.get_total_uploaded()
            download_bytes = self.bandwidth_tracker.get_total_downloaded()
            upload_rate = self.bandwidth_tracker.get_upload_rate()

        # Latency metrics
        inference_ms = 0.0
        stft_ms = 0.0
        training_ms = 0.0
        if self.latency_measure:
            inference_ms = self.latency_measure.get_average("inference")
            stft_ms = self.latency_measure.get_average("stft")
            training_ms = self.latency_measure.get_average("training")

        # System metrics
        cpu_percent, memory_percent, temperature = self._get_system_stats()

        return MetricsSnapshot(
            timestamp=time.time(),
            node_id=self.node_id,
            power_mw=power_mw,
            energy_mwh=energy_mwh,
            upload_bytes=upload_bytes,
            download_bytes=download_bytes,
            upload_rate_bps=upload_rate,
            inference_ms=inference_ms,
            stft_ms=stft_ms,
            training_ms=training_ms,
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            temperature_c=temperature,
            strategy=self.strategy,
            samples_processed=self.samples_processed,
            training_rounds=self.training_rounds
        )

    def _get_system_stats(self) -> tuple:
        """Get system CPU, memory, and temperature."""
        cpu_percent = 0.0
        memory_percent = 0.0
        temperature = 0.0

        if PSUTIL_AVAILABLE:
            try:
                cpu_percent = psutil.cpu_percent()
                memory_percent = psutil.virtual_memory().percent

                # Get temperature (Raspberry Pi specific)
                temps = psutil.sensors_temperatures()
                if 'cpu_thermal' in temps:
                    temperature = temps['cpu_thermal'][0].current
                elif 'coretemp' in temps:
                    temperature = temps['coretemp'][0].current
            except Exception:
                pass

        return cpu_percent, memory_percent, temperature

    def report(self) -> MetricsSnapshot:
        """
        Collect and report current metrics.

        Returns:
            Current metrics snapshot
        """
        snapshot = self.collect_snapshot()
        self.snapshots.append(snapshot)

        # Log summary
        self.logger.info(
            f"Metrics [{self.node_id}]: "
            f"power={snapshot.power_mw:.0f}mW, "
            f"upload={snapshot.upload_bytes/1024:.1f}KB, "
            f"cpu={snapshot.cpu_percent:.0f}%, "
            f"temp={snapshot.temperature_c:.1f}°C"
        )

        # Send via MQTT if available
        if self.mqtt_client:
            topic = f"thesis/edge/{self.node_id}/metrics"
            self.mqtt_client.publish(topic, asdict(snapshot), qos=0)

        return snapshot

    def save_csv(self, filename: Optional[str] = None) -> str:
        """
        Save all snapshots to CSV file.

        Args:
            filename: Output filename (default: auto-generated)

        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"metrics_{self.node_id}_{timestamp}.csv"

        filepath = os.path.join(self.output_dir, filename)

        if not self.snapshots:
            self.logger.warning("No snapshots to save")
            return filepath

        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(asdict(self.snapshots[0]).keys())
            )
            writer.writeheader()
            for snapshot in self.snapshots:
                writer.writerow(asdict(snapshot))

        self.logger.info(f"Saved {len(self.snapshots)} snapshots to {filepath}")
        return filepath

    def save_json(self, filename: Optional[str] = None) -> str:
        """Save all snapshots to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"metrics_{self.node_id}_{timestamp}.json"

        filepath = os.path.join(self.output_dir, filename)

        data = {
            'node_id': self.node_id,
            'strategy': self.strategy,
            'snapshots': [asdict(s) for s in self.snapshots]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        self.logger.info(f"Saved metrics to {filepath}")
        return filepath

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics across all snapshots."""
        if not self.snapshots:
            return {}

        # Aggregate statistics
        powers = [s.power_mw for s in self.snapshots]
        cpus = [s.cpu_percent for s in self.snapshots]

        return {
            'node_id': self.node_id,
            'strategy': self.strategy,
            'duration_s': self.snapshots[-1].timestamp - self.snapshots[0].timestamp,
            'snapshots_count': len(self.snapshots),
            'avg_power_mw': sum(powers) / len(powers),
            'max_power_mw': max(powers),
            'total_energy_mwh': self.snapshots[-1].energy_mwh,
            'total_upload_bytes': self.snapshots[-1].upload_bytes,
            'avg_cpu_percent': sum(cpus) / len(cpus),
            'samples_processed': self.samples_processed,
            'training_rounds': self.training_rounds
        }

    def update_counters(
        self,
        samples: int = 0,
        training_rounds: int = 0
    ) -> None:
        """Update strategy counters."""
        self.samples_processed += samples
        self.training_rounds += training_rounds

    def set_strategy(self, strategy: str) -> None:
        """Set current strategy name."""
        self.strategy = strategy

    def clear(self) -> None:
        """Clear all snapshots."""
        self.snapshots.clear()
