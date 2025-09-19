# File: src/geoworkflow/utils/resource_utils.py
"""
Resource management utilities for the geoworkflow package.

This module provides utilities for managing temporary directories,
file resources, and cleanup operations.
"""

import os
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Optional, Union, Dict, Any, List
from contextlib import contextmanager
import logging
import atexit
from datetime import datetime

from ..core.exceptions import GeoWorkflowError

logger = logging.getLogger(__name__)

# Global registry for resource cleanup
_TEMP_DIRECTORIES: set = set()
_CLEANUP_REGISTERED = False


class ResourceManager:
    """
    Context manager for handling temporary resources and cleanup.
    
    This class provides a clean interface for managing temporary directories,
    files, and other resources that need cleanup after processing.
    """
    
    def __init__(self, prefix: str = "geoworkflow_", cleanup_on_exit: bool = True):
        """
        Initialize the resource manager.
        
        Args:
            prefix: Prefix for temporary directory names
            cleanup_on_exit: Whether to register cleanup on program exit
        """
        self.prefix = prefix
        self.cleanup_on_exit = cleanup_on_exit
        self.temp_dirs: List[Path] = []
        self.temp_files: List[Path] = []
        self._lock = threading.Lock()
        
        # Register for cleanup on exit if requested
        if cleanup_on_exit:
            self._register_cleanup()
    
    def create_temp_directory(self, suffix: str = "") -> Path:
        """
        Create a temporary directory.
        
        Args:
            suffix: Optional suffix for directory name
            
        Returns:
            Path to the created temporary directory
        """
        with self._lock:
            temp_dir = Path(tempfile.mkdtemp(prefix=self.prefix, suffix=suffix))
            self.temp_dirs.append(temp_dir)
            
            # Add to global registry for emergency cleanup
            _TEMP_DIRECTORIES.add(str(temp_dir))
            
            logger.debug(f"Created temporary directory: {temp_dir}")
            return temp_dir
    
    def create_temp_file(self, suffix: str = "", dir: Optional[Path] = None) -> Path:
        """
        Create a temporary file.
        
        Args:
            suffix: File suffix (e.g., '.tif')
            dir: Directory to create file in (None for system temp)
            
        Returns:
            Path to the created temporary file
        """
        with self._lock:
            fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix=self.prefix, dir=dir)
            os.close(fd)  # Close the file descriptor, we just want the path
            
            temp_file = Path(temp_path)
            self.temp_files.append(temp_file)
            
            logger.debug(f"Created temporary file: {temp_file}")
            return temp_file
    
    def cleanup(self):
        """Clean up all managed resources."""
        with self._lock:
            # Clean up temporary files
            for temp_file in self.temp_files:
                try:
                    if temp_file.exists():
                        temp_file.unlink()
                        logger.debug(f"Cleaned up temporary file: {temp_file}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary file {temp_file}: {e}")
            
            # Clean up temporary directories
            for temp_dir in self.temp_dirs:
                try:
                    if temp_dir.exists():
                        shutil.rmtree(temp_dir)
                        logger.debug(f"Cleaned up temporary directory: {temp_dir}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary directory {temp_dir}: {e}")
            
            # Clear lists
            self.temp_files.clear()
            self.temp_dirs.clear()
    
    def _register_cleanup(self):
        """Register cleanup function to run on program exit."""
        global _CLEANUP_REGISTERED
        if not _CLEANUP_REGISTERED:
            atexit.register(_emergency_cleanup)
            _CLEANUP_REGISTERED = True
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


@contextmanager
def temp_directory(prefix: str = "geoworkflow_", suffix: str = ""):
    """
    Context manager for creating a temporary directory.
    
    Args:
        prefix: Prefix for directory name
        suffix: Suffix for directory name
        
    Yields:
        Path to temporary directory
    """
    temp_dir = Path(tempfile.mkdtemp(prefix=prefix, suffix=suffix))
    _TEMP_DIRECTORIES.add(str(temp_dir))
    
    try:
        logger.debug(f"Created temporary directory: {temp_dir}")
        yield temp_dir
    finally:
        try:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
                logger.debug(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean up temporary directory {temp_dir}: {e}")


@contextmanager
def temp_file(suffix: str = "", dir: Optional[Path] = None, prefix: str = "geoworkflow_"):
    """
    Context manager for creating a temporary file.
    
    Args:
        suffix: File suffix
        dir: Directory to create file in
        prefix: Prefix for file name
        
    Yields:
        Path to temporary file
    """
    fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix=prefix, dir=dir)
    os.close(fd)
    
    temp_file_path = Path(temp_path)
    
    try:
        logger.debug(f"Created temporary file: {temp_file_path}")
        yield temp_file_path
    finally:
        try:
            if temp_file_path.exists():
                temp_file_path.unlink()
                logger.debug(f"Cleaned up temporary file: {temp_file_path}")
        except Exception as e:
            logger.warning(f"Failed to clean up temporary file {temp_file_path}: {e}")


def ensure_directory(path: Union[str, Path], create: bool = True) -> Path:
    """
    Ensure a directory exists.
    
    Args:
        path: Directory path
        create: Whether to create if it doesn't exist
        
    Returns:
        Path object
        
    Raises:
        GeoWorkflowError: If directory doesn't exist and create=False
    """
    path_obj = Path(path)
    
    if not path_obj.exists():
        if create:
            path_obj.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {path_obj}")
        else:
            raise GeoWorkflowError(f"Directory does not exist: {path_obj}")
    
    return path_obj


def safe_remove_file(file_path: Union[str, Path]) -> bool:
    """
    Safely remove a file.
    
    Args:
        file_path: Path to file to remove
        
    Returns:
        True if removed successfully, False otherwise
    """
    try:
        file_path = Path(file_path)
        if file_path.exists():
            file_path.unlink()
            logger.debug(f"Removed file: {file_path}")
            return True
        return False
    except Exception as e:
        logger.warning(f"Failed to remove file {file_path}: {e}")
        return False


def safe_remove_directory(dir_path: Union[str, Path]) -> bool:
    """
    Safely remove a directory and its contents.
    
    Args:
        dir_path: Path to directory to remove
        
    Returns:
        True if removed successfully, False otherwise
    """
    try:
        dir_path = Path(dir_path)
        if dir_path.exists():
            shutil.rmtree(dir_path)
            logger.debug(f"Removed directory: {dir_path}")
            return True
        return False
    except Exception as e:
        logger.warning(f"Failed to remove directory {dir_path}: {e}")
        return False


def get_safe_filename(name: str, max_length: int = 255) -> str:
    """
    Convert a string to a safe filename.
    
    Args:
        name: Original name
        max_length: Maximum filename length
        
    Returns:
        Safe filename string
    """
    # Replace problematic characters
    safe_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
    safe_name = "".join(c if c in safe_chars else "_" for c in name)
    
    # Ensure not too long
    if len(safe_name) > max_length:
        safe_name = safe_name[:max_length]
    
    # Ensure doesn't start with dot or dash
    if safe_name.startswith(('.', '-')):
        safe_name = 'file_' + safe_name
    
    return safe_name


def get_disk_usage(path: Union[str, Path]) -> Dict[str, int]:
    """
    Get disk usage statistics for a path.
    
    Args:
        path: Path to check
        
    Returns:
        Dictionary with 'total', 'used', 'free' in bytes
    """
    try:
        path = Path(path)
        stat = shutil.disk_usage(path)
        return {
            'total': stat.total,
            'used': stat.total - stat.free,
            'free': stat.free
        }
    except Exception as e:
        logger.warning(f"Failed to get disk usage for {path}: {e}")
        return {'total': 0, 'used': 0, 'free': 0}


def check_disk_space(path: Union[str, Path], required_bytes: int) -> bool:
    """
    Check if sufficient disk space is available.
    
    Args:
        path: Path to check
        required_bytes: Required space in bytes
        
    Returns:
        True if sufficient space available
    """
    usage = get_disk_usage(path)
    return usage['free'] >= required_bytes


def format_bytes(bytes_value: int) -> str:
    """
    Format bytes as human-readable string.
    
    Args:
        bytes_value: Number of bytes
        
    Returns:
        Formatted string (e.g., "1.5 GB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"


def _emergency_cleanup():
    """Emergency cleanup function called on program exit."""
    logger.debug("Running emergency cleanup of temporary directories")
    
    for temp_dir_str in list(_TEMP_DIRECTORIES):
        try:
            temp_dir = Path(temp_dir_str)
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
                logger.debug(f"Emergency cleanup: removed {temp_dir}")
        except Exception as e:
            logger.warning(f"Emergency cleanup failed for {temp_dir_str}: {e}")


class ProcessingMetrics:
    """
    Simple metrics collection for processing operations.
    """
    
    def __init__(self):
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.files_processed: int = 0
        self.files_failed: int = 0
        self.files_skipped: int = 0
        self.bytes_processed: int = 0
        self.custom_metrics: Dict[str, Any] = {}
    
    def start(self):
        """Mark the start of processing."""
        self.start_time = datetime.now()
    
    def finish(self):
        """Mark the end of processing."""
        self.end_time = datetime.now()
    
    @property
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
    
    @property
    def total_files(self) -> int:
        """Get total number of files."""
        return self.files_processed + self.files_failed + self.files_skipped
    
    def add_metric(self, name: str, value: Any):
        """Add a custom metric."""
        self.custom_metrics[name] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'elapsed_time': self.elapsed_time,
            'files_processed': self.files_processed,
            'files_failed': self.files_failed,
            'files_skipped': self.files_skipped,
            'total_files': self.total_files,
            'bytes_processed': self.bytes_processed,
            'bytes_formatted': format_bytes(self.bytes_processed),
            'custom_metrics': self.custom_metrics
        }
