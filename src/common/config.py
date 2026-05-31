"""
Configuration Module
====================

Handles loading and validation of YAML configuration files.

Usage:
    config = load_config("configs/edge_config.yaml")
    validate_config(config, "edge")
"""

import os
import logging
from typing import Dict, Any, Optional

# YAML support
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


def load_config(path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        path: Path to YAML configuration file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If YAML parsing fails
    """
    if not YAML_AVAILABLE:
        raise RuntimeError("PyYAML not installed. Run: pip install pyyaml")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to parse config: {e}")

    return config or {}


def validate_config(config: Dict[str, Any], config_type: str) -> bool:
    """
    Validate configuration against expected schema.

    Args:
        config: Configuration dictionary
        config_type: Type of config ("edge" or "server")

    Returns:
        True if valid

    Raises:
        ValueError: If validation fails
    """
    if config_type == "edge":
        return _validate_edge_config(config)
    elif config_type == "server":
        return _validate_server_config(config)
    else:
        raise ValueError(f"Unknown config type: {config_type}")


def _validate_edge_config(config: Dict[str, Any]) -> bool:
    """Validate edge configuration."""
    required_sections = ["node", "server", "strategy"]

    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section: {section}")

    # Validate node
    if "id" not in config.get("node", {}):
        raise ValueError("node.id is required")

    # Validate server
    if "ip" not in config.get("server", {}):
        raise ValueError("server.ip is required")

    # Validate strategy
    valid_strategies = ["centralized", "hybrid", "federated"]
    strategy_type = config.get("strategy", {}).get("type")
    if strategy_type not in valid_strategies:
        raise ValueError(
            f"Invalid strategy type: {strategy_type}. "
            f"Must be one of: {valid_strategies}"
        )

    return True


def _validate_server_config(config: Dict[str, Any]) -> bool:
    """Validate server configuration."""
    required_sections = ["server"]

    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section: {section}")

    # Validate mode
    valid_modes = ["centralized", "federated"]
    mode = config.get("server", {}).get("mode")
    if mode not in valid_modes:
        raise ValueError(
            f"Invalid server mode: {mode}. "
            f"Must be one of: {valid_modes}"
        )

    return True


def get_config_value(
    config: Dict[str, Any],
    key_path: str,
    default: Any = None
) -> Any:
    """
    Get nested config value using dot notation.

    Args:
        config: Configuration dictionary
        key_path: Dot-separated path (e.g., "server.mqtt.port")
        default: Default value if not found

    Returns:
        Configuration value or default
    """
    keys = key_path.split(".")
    value = config

    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default

    return value


def merge_configs(
    base: Dict[str, Any],
    override: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Merge two configs, with override taking precedence.

    Args:
        base: Base configuration
        override: Override configuration

    Returns:
        Merged configuration
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value

    return result
