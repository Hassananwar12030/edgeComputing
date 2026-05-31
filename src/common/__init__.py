"""
Common Module
=============

Shared utilities and constants used by both edge and server.

Contents:
    - config: Configuration loading and validation
    - constants: Project-wide constants
    - utils: Utility functions
    - serialization: Data serialization helpers
"""

from .config import load_config, validate_config
from .constants import *
from .utils import setup_logging, get_timestamp

__all__ = [
    "load_config",
    "validate_config",
    "setup_logging",
    "get_timestamp"
]
