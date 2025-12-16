"""
Comprehensive utility module for CAMAC-DRA project.

This module provides utility functions and classes for:
- Logging configuration and management
- Metrics tracking and aggregation
- Configuration management
- Mathematical utilities
- Performance monitoring

Created: 2025-12-16
Author: muddsairsharif
"""

import logging
import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
import functools
import time
from pathlib import Path
import math
from enum import Enum


# ============================================================================
# LOGGING UTILITIES
# ============================================================================

class LogLevel(Enum):
    """Enumeration for logging levels."""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class LoggerConfig:
    """Configuration class for logger setup."""
    
    def __init__(
        self,
        name: str,
        level: LogLevel = LogLevel.INFO,
        log_file: Optional[str] = None,
        format_string: Optional[str] = None,
    ):
        """
        Initialize logger configuration.
        
        Args:
            name: Logger name
            level: Logging level
            log_file: Optional log file path
            format_string: Optional custom format string
        """
        self.name = name
        self.level = level
        self.log_file = log_file
        self.format_string = (
            format_string or
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    
    def create_logger(self) -> logging.Logger:
        """
        Create and configure a logger instance.
        
        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(self.name)
        logger.setLevel(self.level.value)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.level.value)
        formatter = logging.Formatter(self.format_string)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler (if specified)
        if self.log_file:
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setLevel(self.level.value)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger


def get_logger(name: str, **kwargs) -> logging.Logger:
    """
    Get or create a logger with the given name.
    
    Args:
        name: Logger name
        **kwargs: Additional arguments passed to LoggerConfig
    
    Returns:
        Configured logger instance
    """
    config = LoggerConfig(name, **kwargs)
    return config.create_logger()


# ============================================================================
# METRICS TRACKING
# ============================================================================

@dataclass
class Metric:
    """Data class for storing metric information."""
    name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags,
        }


class MetricsTracker:
    """Track and aggregate metrics."""
    
    def __init__(self, name: str = "MetricsTracker"):
        """
        Initialize metrics tracker.
        
        Args:
            name: Tracker name
        """
        self.name = name
        self.metrics: Dict[str, List[Metric]] = {}
        self.logger = get_logger(f"{self.name}.MetricsTracker")
    
    def record_metric(
        self,
        metric_name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Record a metric value.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            tags: Optional tags for the metric
        """
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        
        metric = Metric(
            name=metric_name,
            value=value,
            tags=tags or {},
        )
        self.metrics[metric_name].append(metric)
        self.logger.debug(f"Recorded metric: {metric_name}={value}")
    
    def get_metric_stats(self, metric_name: str) -> Optional[Dict[str, float]]:
        """
        Get statistics for a metric.
        
        Args:
            metric_name: Name of the metric
        
        Returns:
            Dictionary with min, max, mean, and count statistics
        """
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return None
        
        values = [m.value for m in self.metrics[metric_name]]
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
            "sum": sum(values),
        }
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """
        Get all recorded metrics.
        
        Returns:
            Dictionary of all metrics organized by name
        """
        return {
            name: [m.to_dict() for m in metrics]
            for name, metrics in self.metrics.items()
        }
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics.clear()
        self.logger.info("Metrics tracker reset")


# ============================================================================
# CONFIGURATION MANAGEMENT
# ============================================================================

@dataclass
class ConfigValue:
    """Represents a configuration value."""
    key: str
    value: Any
    type_hint: type = str
    description: Optional[str] = None
    required: bool = False


class ConfigManager:
    """Manage application configuration."""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_file: Optional path to JSON config file
        """
        self.config: Dict[str, Any] = {}
        self.logger = get_logger("ConfigManager")
        
        if config_file and os.path.exists(config_file):
            self.load_from_file(config_file)
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value.
        
        Args:
            key: Configuration key
            value: Configuration value
        """
        self.config[key] = value
        self.logger.debug(f"Config set: {key}={value}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
        
        Returns:
            Configuration value or default
        """
        return self.config.get(key, default)
    
    def get_or_raise(self, key: str) -> Any:
        """
        Get a configuration value or raise KeyError.
        
        Args:
            key: Configuration key
        
        Returns:
            Configuration value
        
        Raises:
            KeyError: If key not found
        """
        if key not in self.config:
            raise KeyError(f"Configuration key '{key}' not found")
        return self.config[key]
    
    def load_from_file(self, config_file: str) -> None:
        """
        Load configuration from JSON file.
        
        Args:
            config_file: Path to JSON config file
        """
        try:
            with open(config_file, 'r') as f:
                self.config.update(json.load(f))
            self.logger.info(f"Configuration loaded from {config_file}")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.logger.error(f"Failed to load config file: {e}")
            raise
    
    def save_to_file(self, config_file: str) -> None:
        """
        Save configuration to JSON file.
        
        Args:
            config_file: Path to JSON config file
        """
        try:
            os.makedirs(os.path.dirname(config_file) or ".", exist_ok=True)
            with open(config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            self.logger.info(f"Configuration saved to {config_file}")
        except IOError as e:
            self.logger.error(f"Failed to save config file: {e}")
            raise
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Get configuration as dictionary.
        
        Returns:
            Configuration dictionary
        """
        return self.config.copy()
    
    def clear(self) -> None:
        """Clear all configuration."""
        self.config.clear()


# ============================================================================
# MATHEMATICAL UTILITIES
# ============================================================================

class MathUtils:
    """Collection of mathematical utility functions."""
    
    @staticmethod
    def normalize(value: float, min_val: float, max_val: float) -> float:
        """
        Normalize a value to range [0, 1].
        
        Args:
            value: Value to normalize
            min_val: Minimum value of range
            max_val: Maximum value of range
        
        Returns:
            Normalized value
        """
        if max_val == min_val:
            return 0.0
        return (value - min_val) / (max_val - min_val)
    
    @staticmethod
    def denormalize(
        normalized_value: float,
        min_val: float,
        max_val: float,
    ) -> float:
        """
        Denormalize a value from range [0, 1] to original range.
        
        Args:
            normalized_value: Normalized value
            min_val: Minimum value of original range
            max_val: Maximum value of original range
        
        Returns:
            Denormalized value
        """
        return normalized_value * (max_val - min_val) + min_val
    
    @staticmethod
    def clamp(value: float, min_val: float, max_val: float) -> float:
        """
        Clamp a value to a range.
        
        Args:
            value: Value to clamp
            min_val: Minimum value
            max_val: Maximum value
        
        Returns:
            Clamped value
        """
        return max(min_val, min(value, max_val))
    
    @staticmethod
    def lerp(a: float, b: float, t: float) -> float:
        """
        Linear interpolation between two values.
        
        Args:
            a: Start value
            b: End value
            t: Interpolation factor (0 to 1)
        
        Returns:
            Interpolated value
        """
        return a + (b - a) * t
    
    @staticmethod
    def distance(p1: Tuple[float, ...], p2: Tuple[float, ...]) -> float:
        """
        Calculate Euclidean distance between two points.
        
        Args:
            p1: First point
            p2: Second point
        
        Returns:
            Euclidean distance
        """
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(p1, p2)))
    
    @staticmethod
    def mean(values: List[float]) -> float:
        """
        Calculate mean of values.
        
        Args:
            values: List of values
        
        Returns:
            Mean value
        """
        return sum(values) / len(values) if values else 0.0
    
    @staticmethod
    def median(values: List[float]) -> float:
        """
        Calculate median of values.
        
        Args:
            values: List of values
        
        Returns:
            Median value
        """
        sorted_values = sorted(values)
        n = len(sorted_values)
        if n == 0:
            return 0.0
        if n % 2 == 1:
            return sorted_values[n // 2]
        return (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2
    
    @staticmethod
    def std_dev(values: List[float]) -> float:
        """
        Calculate standard deviation of values.
        
        Args:
            values: List of values
        
        Returns:
            Standard deviation
        """
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return math.sqrt(variance)


# ============================================================================
# PERFORMANCE MONITORING
# ============================================================================

class TimerContext:
    """Context manager for timing code execution."""
    
    def __init__(self, name: str, logger: Optional[logging.Logger] = None):
        """
        Initialize timer context.
        
        Args:
            name: Timer name
            logger: Optional logger instance
        """
        self.name = name
        self.logger = logger or get_logger("TimerContext")
        self.start_time = None
        self.elapsed_time = None
    
    def __enter__(self):
        """Enter context."""
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        self.elapsed_time = time.time() - self.start_time
        self.logger.info(
            f"{self.name} completed in {self.elapsed_time:.4f} seconds"
        )
    
    def get_elapsed(self) -> float:
        """
        Get elapsed time in seconds.
        
        Returns:
            Elapsed time
        """
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time


def timing_decorator(func):
    """
    Decorator to measure function execution time.
    
    Args:
        func: Function to decorate
    
    Returns:
        Decorated function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        with TimerContext(f"{func.__name__}", logger):
            return func(*args, **kwargs)
    return wrapper


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def validate_input(value: Any, expected_type: type) -> bool:
    """
    Validate input type.
    
    Args:
        value: Value to validate
        expected_type: Expected type
    
    Returns:
        True if value is of expected type
    """
    return isinstance(value, expected_type)


def safe_divide(
    numerator: float,
    denominator: float,
    default: float = 0.0,
) -> float:
    """
    Safely divide two numbers.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if denominator is zero
    
    Returns:
        Result of division or default value
    """
    return numerator / denominator if denominator != 0 else default


def flatten_dict(
    d: Dict[str, Any],
    parent_key: str = "",
    sep: str = ".",
) -> Dict[str, Any]:
    """
    Flatten a nested dictionary.
    
    Args:
        d: Dictionary to flatten
        parent_key: Parent key prefix
        sep: Separator for keys
    
    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(
                flatten_dict(v, new_key, sep=sep).items()
            )
        else:
            items.append((new_key, v))
    return dict(items)


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Logging
    "LogLevel",
    "LoggerConfig",
    "get_logger",
    # Metrics
    "Metric",
    "MetricsTracker",
    # Configuration
    "ConfigValue",
    "ConfigManager",
    # Math utilities
    "MathUtils",
    # Performance monitoring
    "TimerContext",
    "timing_decorator",
    # Utility functions
    "validate_input",
    "safe_divide",
    "flatten_dict",
]
