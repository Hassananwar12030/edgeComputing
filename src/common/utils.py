"""
Utility Functions
=================

Common utility functions used throughout the project.
"""

import os
import sys
import time
import logging
from datetime import datetime
from typing import Optional, Dict, Any
import json


def setup_logging(
    name: str = "thesis",
    level: str = "INFO",
    log_file: Optional[str] = None,
    console: bool = True
) -> logging.Logger:
    """
    Setup logging configuration.

    Args:
        name: Logger name
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_file: Path to log file (None for no file)
        console: Enable console output

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    # Format
    formatter = logging.Formatter(
        '[%(asctime)s] [%(name)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        # Ensure directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_timestamp() -> float:
    """Get current timestamp."""
    return time.time()


def get_datetime_string(fmt: str = "%Y-%m-%d_%H-%M-%S") -> str:
    """Get formatted datetime string."""
    return datetime.now().strftime(fmt)


def ensure_dir(path: str) -> str:
    """
    Ensure directory exists, create if needed.

    Args:
        path: Directory path

    Returns:
        The same path
    """
    os.makedirs(path, exist_ok=True)
    return path


def get_file_size_mb(path: str) -> float:
    """Get file size in megabytes."""
    if os.path.exists(path):
        return os.path.getsize(path) / (1024 * 1024)
    return 0.0


def get_dir_size_mb(path: str) -> float:
    """Get total directory size in megabytes."""
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total += os.path.getsize(fp)
    return total / (1024 * 1024)


def save_json(data: Dict[str, Any], path: str, indent: int = 2) -> None:
    """Save dictionary to JSON file."""
    ensure_dir(os.path.dirname(path))
    with open(path, 'w') as f:
        json.dump(data, f, indent=indent, default=str)


def load_json(path: str) -> Dict[str, Any]:
    """Load dictionary from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def format_bytes(size: int) -> str:
    """Format bytes to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} PB"


def format_duration(seconds: float) -> str:
    """Format duration to human readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


class Timer:
    """Simple timer context manager."""

    def __init__(self, name: str = "operation"):
        self.name = name
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end_time = time.perf_counter()
        return False

    @property
    def elapsed(self) -> float:
        """Elapsed time in seconds."""
        if self.end_time:
            return self.end_time - self.start_time
        elif self.start_time:
            return time.perf_counter() - self.start_time
        return 0.0

    @property
    def elapsed_ms(self) -> float:
        """Elapsed time in milliseconds."""
        return self.elapsed * 1000


def is_raspberry_pi() -> bool:
    """Check if running on Raspberry Pi."""
    try:
        with open('/proc/device-tree/model', 'r') as f:
            model = f.read()
            return 'raspberry pi' in model.lower()
    except:
        return False


def get_system_info() -> Dict[str, Any]:
    """Get system information."""
    import platform

    info = {
        'platform': platform.system(),
        'platform_release': platform.release(),
        'platform_version': platform.version(),
        'architecture': platform.machine(),
        'processor': platform.processor(),
        'python_version': platform.python_version(),
        'is_raspberry_pi': is_raspberry_pi()
    }

    try:
        import psutil
        info['cpu_count'] = psutil.cpu_count()
        info['memory_total_gb'] = psutil.virtual_memory().total / (1024**3)
    except ImportError:
        pass

    return info
