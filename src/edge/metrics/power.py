"""
Power Monitoring Module
=======================

Measures power consumption using INA219 current sensor.

The INA219 is connected via I2C and measures:
- Bus voltage (V)
- Shunt voltage (mV)
- Current (mA)
- Power (mW)

Wiring (Raspberry Pi):
    INA219 VCC → Pi 3.3V
    INA219 GND → Pi GND
    INA219 SDA → Pi GPIO 2 (SDA)
    INA219 SCL → Pi GPIO 3 (SCL)
    INA219 VIN+ → Power input +
    INA219 VIN- → Pi 5V
    INA219 GND  → Power input -

Usage:
    monitor = PowerMonitor()
    monitor.start()
    # ... do work ...
    avg_power = monitor.get_average_power()
    monitor.stop()
"""

import logging
import time
import threading
from typing import Optional, Dict, Any, List
from collections import deque

# INA219 library
try:
    from ina219 import INA219
    from ina219 import DeviceRangeError
    INA219_AVAILABLE = True
except ImportError:
    INA219_AVAILABLE = False


class PowerMonitor:
    """
    Power consumption monitor using INA219.

    Continuously samples power consumption in a background thread
    and provides aggregate statistics.

    Attributes:
        sample_interval: Seconds between samples
        measurements: Recent power measurements (mW)
    """

    # INA219 configuration
    SHUNT_OHMS = 0.1  # 0.1 ohm shunt resistor
    MAX_EXPECTED_AMPS = 3.0  # Max expected current

    def __init__(
        self,
        address: int = 0x40,
        sample_interval: float = 0.1,
        buffer_size: int = 1000
    ):
        """
        Initialize power monitor.

        Args:
            address: I2C address of INA219 (default 0x40)
            sample_interval: Seconds between samples
            buffer_size: Number of samples to keep in buffer
        """
        self.address = address
        self.sample_interval = sample_interval
        self.buffer_size = buffer_size

        self.ina = None
        self.measurements: deque = deque(maxlen=buffer_size)
        self._sampling_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.is_running = False

        self.logger = logging.getLogger(__name__)

        # Mock mode for development
        self._mock_mode = not INA219_AVAILABLE

        if self._mock_mode:
            self.logger.warning(
                "INA219 not available - running in mock mode"
            )

    def start(self) -> bool:
        """
        Start power monitoring.

        Returns:
            True if started successfully
        """
        if self.is_running:
            return True

        if not self._mock_mode:
            try:
                self.ina = INA219(
                    self.SHUNT_OHMS,
                    self.MAX_EXPECTED_AMPS,
                    address=self.address
                )
                self.ina.configure()
                self.logger.info("INA219 initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize INA219: {e}")
                self._mock_mode = True

        # Start sampling thread
        self._stop_event.clear()
        self._sampling_thread = threading.Thread(
            target=self._sampling_loop,
            daemon=True
        )
        self._sampling_thread.start()
        self.is_running = True

        self.logger.info("Power monitoring started")
        return True

    def stop(self) -> None:
        """Stop power monitoring."""
        self._stop_event.set()
        if self._sampling_thread:
            self._sampling_thread.join(timeout=2.0)
        self.is_running = False
        self.logger.info("Power monitoring stopped")

    def _sampling_loop(self) -> None:
        """Background sampling loop."""
        while not self._stop_event.is_set():
            try:
                power = self._read_power()
                self.measurements.append({
                    'power_mw': power,
                    'timestamp': time.time()
                })
            except Exception as e:
                self.logger.debug(f"Sampling error: {e}")

            time.sleep(self.sample_interval)

    def _read_power(self) -> float:
        """Read current power in mW."""
        if self._mock_mode:
            # Simulate Pi power consumption (2-5W typically)
            import random
            return 2500 + random.random() * 2500

        try:
            return self.ina.power()
        except DeviceRangeError:
            self.logger.warning("INA219 device range error")
            return 0.0

    def get_current_power(self) -> float:
        """Get most recent power reading in mW."""
        if self.measurements:
            return self.measurements[-1]['power_mw']
        return 0.0

    def get_average_power(
        self,
        duration: Optional[float] = None
    ) -> float:
        """
        Get average power over duration.

        Args:
            duration: Seconds to average (None = all samples)

        Returns:
            Average power in mW
        """
        if not self.measurements:
            return 0.0

        if duration is None:
            samples = list(self.measurements)
        else:
            cutoff = time.time() - duration
            samples = [m for m in self.measurements if m['timestamp'] > cutoff]

        if not samples:
            return 0.0

        return sum(m['power_mw'] for m in samples) / len(samples)

    def get_energy_consumed(
        self,
        duration: Optional[float] = None
    ) -> float:
        """
        Get energy consumed in mWh.

        Args:
            duration: Seconds to calculate (None = all samples)

        Returns:
            Energy in mWh
        """
        if not self.measurements:
            return 0.0

        if duration is None:
            samples = list(self.measurements)
        else:
            cutoff = time.time() - duration
            samples = [m for m in self.measurements if m['timestamp'] > cutoff]

        if len(samples) < 2:
            return 0.0

        # Integrate power over time
        total_energy_mws = 0.0
        for i in range(1, len(samples)):
            dt = samples[i]['timestamp'] - samples[i-1]['timestamp']
            avg_power = (samples[i]['power_mw'] + samples[i-1]['power_mw']) / 2
            total_energy_mws += avg_power * dt

        # Convert mW*s to mWh
        return total_energy_mws / 3600

    def get_stats(self) -> Dict[str, float]:
        """Get power statistics."""
        if not self.measurements:
            return {}

        powers = [m['power_mw'] for m in self.measurements]

        return {
            'current_mw': powers[-1] if powers else 0,
            'average_mw': sum(powers) / len(powers),
            'min_mw': min(powers),
            'max_mw': max(powers),
            'samples': len(powers)
        }

    def clear(self) -> None:
        """Clear measurement buffer."""
        self.measurements.clear()
