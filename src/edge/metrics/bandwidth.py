"""
Bandwidth Tracking Module
=========================

Tracks network bandwidth usage for data uploads.

Important for comparing strategies:
- Strategy A: Highest bandwidth (raw audio)
- Strategy B: Medium bandwidth (spectrograms)
- Strategy C: Lowest bandwidth (model weights only)

Methods:
1. Track bytes sent/received via application
2. Monitor network interface statistics
3. Calculate rolling averages

Usage:
    tracker = BandwidthTracker()
    tracker.record_upload(1024)  # Record 1KB upload
    stats = tracker.get_stats()
"""

import logging
import time
import threading
from typing import Dict, Any, Optional
from collections import deque
from dataclasses import dataclass

# psutil for system network stats
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


@dataclass
class TransferRecord:
    """Record of a data transfer."""
    bytes_count: int
    timestamp: float
    direction: str  # 'upload' or 'download'
    category: str   # 'data', 'weights', 'metrics', etc.


class BandwidthTracker:
    """
    Network bandwidth tracker.

    Tracks both application-level transfers and system-level
    network interface statistics.

    Attributes:
        records: Recent transfer records
        interface: Network interface to monitor (e.g., 'wlan0')
    """

    def __init__(
        self,
        interface: Optional[str] = None,
        buffer_size: int = 1000
    ):
        """
        Initialize bandwidth tracker.

        Args:
            interface: Network interface name (None for auto-detect)
            buffer_size: Number of records to keep
        """
        self.interface = interface or self._detect_interface()
        self.buffer_size = buffer_size

        self.records: deque = deque(maxlen=buffer_size)
        self._baseline_sent = 0
        self._baseline_recv = 0
        self._baseline_time = time.time()

        self.logger = logging.getLogger(__name__)

        # Initialize baseline
        if PSUTIL_AVAILABLE:
            self._update_baseline()

    def _detect_interface(self) -> str:
        """Auto-detect primary network interface."""
        if not PSUTIL_AVAILABLE:
            return 'wlan0'

        # Get interface with most traffic
        stats = psutil.net_io_counters(pernic=True)

        # Prefer wireless interfaces
        for name in stats:
            if name.startswith('wlan') or name.startswith('wl'):
                return name

        # Fall back to first non-loopback
        for name in stats:
            if name != 'lo':
                return name

        return 'eth0'

    def _update_baseline(self) -> None:
        """Update baseline network counters."""
        if not PSUTIL_AVAILABLE:
            return

        try:
            stats = psutil.net_io_counters(pernic=True)
            if self.interface in stats:
                self._baseline_sent = stats[self.interface].bytes_sent
                self._baseline_recv = stats[self.interface].bytes_recv
                self._baseline_time = time.time()
        except Exception as e:
            self.logger.debug(f"Failed to update baseline: {e}")

    def record_upload(
        self,
        bytes_count: int,
        category: str = "data"
    ) -> None:
        """
        Record an upload.

        Args:
            bytes_count: Number of bytes uploaded
            category: Transfer category (data, weights, metrics)
        """
        record = TransferRecord(
            bytes_count=bytes_count,
            timestamp=time.time(),
            direction='upload',
            category=category
        )
        self.records.append(record)
        self.logger.debug(f"Recorded upload: {bytes_count} bytes ({category})")

    def record_download(
        self,
        bytes_count: int,
        category: str = "model"
    ) -> None:
        """
        Record a download.

        Args:
            bytes_count: Number of bytes downloaded
            category: Transfer category
        """
        record = TransferRecord(
            bytes_count=bytes_count,
            timestamp=time.time(),
            direction='download',
            category=category
        )
        self.records.append(record)
        self.logger.debug(f"Recorded download: {bytes_count} bytes ({category})")

    def get_total_uploaded(
        self,
        duration: Optional[float] = None,
        category: Optional[str] = None
    ) -> int:
        """
        Get total bytes uploaded.

        Args:
            duration: Time window in seconds (None = all time)
            category: Filter by category (None = all)

        Returns:
            Total bytes uploaded
        """
        cutoff = time.time() - duration if duration else 0

        total = 0
        for record in self.records:
            if record.direction != 'upload':
                continue
            if record.timestamp < cutoff:
                continue
            if category and record.category != category:
                continue
            total += record.bytes_count

        return total

    def get_total_downloaded(
        self,
        duration: Optional[float] = None,
        category: Optional[str] = None
    ) -> int:
        """Get total bytes downloaded."""
        cutoff = time.time() - duration if duration else 0

        total = 0
        for record in self.records:
            if record.direction != 'download':
                continue
            if record.timestamp < cutoff:
                continue
            if category and record.category != category:
                continue
            total += record.bytes_count

        return total

    def get_upload_rate(self, duration: float = 60.0) -> float:
        """
        Get upload rate in bytes per second.

        Args:
            duration: Time window in seconds

        Returns:
            Bytes per second
        """
        total = self.get_total_uploaded(duration)
        return total / duration if duration > 0 else 0

    def get_system_bandwidth(self) -> Dict[str, int]:
        """
        Get system-level bandwidth from interface.

        Returns total bytes sent/received since baseline.
        """
        if not PSUTIL_AVAILABLE:
            return {'sent': 0, 'recv': 0}

        try:
            stats = psutil.net_io_counters(pernic=True)
            if self.interface in stats:
                iface_stats = stats[self.interface]
                return {
                    'sent': iface_stats.bytes_sent - self._baseline_sent,
                    'recv': iface_stats.bytes_recv - self._baseline_recv,
                    'duration': time.time() - self._baseline_time
                }
        except Exception as e:
            self.logger.debug(f"Failed to get system bandwidth: {e}")

        return {'sent': 0, 'recv': 0, 'duration': 0}

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive bandwidth statistics."""
        sys_bw = self.get_system_bandwidth()

        return {
            'app_uploaded_bytes': self.get_total_uploaded(),
            'app_downloaded_bytes': self.get_total_downloaded(),
            'app_upload_rate_bps': self.get_upload_rate(),
            'system_sent_bytes': sys_bw.get('sent', 0),
            'system_recv_bytes': sys_bw.get('recv', 0),
            'interface': self.interface,
            'records_count': len(self.records),
            'by_category': self._get_by_category()
        }

    def _get_by_category(self) -> Dict[str, int]:
        """Get upload totals by category."""
        totals = {}
        for record in self.records:
            if record.direction == 'upload':
                cat = record.category
                totals[cat] = totals.get(cat, 0) + record.bytes_count
        return totals

    def reset(self) -> None:
        """Reset all tracking."""
        self.records.clear()
        self._update_baseline()

    def reset_baseline(self) -> None:
        """Reset just the system baseline."""
        self._update_baseline()
