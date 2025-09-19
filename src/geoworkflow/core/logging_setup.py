"""
Logging setup for the geoworkflow package.

This module provides centralized logging configuration for all
geoworkflow components.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional, Union
from datetime import datetime

from .constants import DEFAULT_LOG_LEVEL, DEFAULT_LOG_FORMAT, LOG_LEVELS


class GeoWorkflowFormatter(logging.Formatter):
    """Custom formatter for geoworkflow logging."""
    
    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None):
        super().__init__(fmt or DEFAULT_LOG_FORMAT, datefmt)
    
    def format(self, record):
        # Add color coding for console output if needed
        if hasattr(record, 'levelname'):
            # Add any custom formatting here
            pass
        return super().format(record)


def setup_logging(level: Union[str, int] = DEFAULT_LOG_LEVEL,
                 log_file: Optional[Union[str, Path]] = None,
                 format_string: Optional[str] = None,
                 enable_console: bool = True) -> logging.Logger:
    """
    Set up logging for the geoworkflow package.
    
    Args:
        level: Logging level (string or int)
        log_file: Optional path to log file
        format_string: Optional custom format string
        enable_console: Whether to enable console logging
        
    Returns:
        Configured logger instance
    """
    # Convert string level to int if needed
    if isinstance(level, str):
        level = LOG_LEVELS.get(level.upper(), logging.INFO)
    
    # Get root logger for geoworkflow
    logger = logging.getLogger('geoworkflow')
    logger.setLevel(level)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = GeoWorkflowFormatter(format_string)
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Prevent propagation to avoid duplicate messages
    logger.propagate = False
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific component.
    
    Args:
        name: Logger name (will be prefixed with 'geoworkflow.')
        
    Returns:
        Logger instance
    """
    if not name.startswith('geoworkflow.'):
        name = f'geoworkflow.{name}'
    
    return logging.getLogger(name)


def configure_processor_logging(processor_name: str,
                              level: Optional[Union[str, int]] = None,
                              log_dir: Optional[Union[str, Path]] = None) -> logging.Logger:
    """
    Configure logging for a specific processor.
    
    Args:
        processor_name: Name of the processor
        level: Optional logging level override
        log_dir: Optional directory for processor-specific log files
        
    Returns:
        Configured logger for the processor
    """
    logger_name = f'geoworkflow.{processor_name}'
    logger = logging.getLogger(logger_name)
    
    if level is not None:
        if isinstance(level, str):
            level = LOG_LEVELS.get(level.upper(), logging.INFO)
        logger.setLevel(level)
    
    # Add processor-specific file handler if log_dir is provided
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_path / f"{processor_name}_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logger.level)
        
        formatter = GeoWorkflowFormatter()
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
    
    return logger


def set_quiet_mode(quiet: bool = True):
    """
    Enable or disable quiet mode (suppress INFO and DEBUG messages).
    
    Args:
        quiet: If True, only show WARNING and above
    """
    logger = logging.getLogger('geoworkflow')
    
    if quiet:
        logger.setLevel(logging.WARNING)
        # Update all handlers
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(logging.WARNING)
    else:
        logger.setLevel(logging.INFO)
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(logging.INFO)


def set_verbose_mode(verbose: bool = True):
    """
    Enable or disable verbose mode (show DEBUG messages).
    
    Args:
        verbose: If True, show all messages including DEBUG
    """
    logger = logging.getLogger('geoworkflow')
    
    if verbose:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
        for handler in logger.handlers:
            handler.setLevel(logging.INFO)


def log_system_info():
    """Log system and environment information."""
    import sys
    import platform
    
    logger = get_logger('system')
    
    logger.info("=== GeoWorkflow System Information ===")
    logger.info(f"Python Version: {sys.version}")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Architecture: {platform.architecture()[0]}")
    
    try:
        from .. import __version__
        logger.info(f"GeoWorkflow Version: {__version__}")
    except ImportError:
        logger.info("GeoWorkflow Version: Unknown")


def create_performance_logger(name: str, log_dir: Optional[Path] = None) -> logging.Logger:
    """
    Create a logger specifically for performance monitoring.
    
    Args:
        name: Logger name
        log_dir: Directory for performance logs
        
    Returns:
        Performance logger
    """
    logger_name = f'geoworkflow.performance.{name}'
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        perf_log_file = log_path / f"performance_{name}.log"
        
        # Performance logs use a simpler format
        perf_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler = logging.FileHandler(perf_log_file)
        file_handler.setFormatter(perf_formatter)
        
        logger.addHandler(file_handler)
        logger.propagate = False
    
    return logger