# File: src/geoworkflow/utils/progress_utils.py
"""
Progress tracking utilities for the geoworkflow package.

This module provides utilities for tracking and displaying progress
during long-running operations.
"""

import time
from typing import Optional, Callable, Any
from datetime import datetime, timedelta
import logging

try:
    from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TaskID
    from rich.console import Console
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

logger = logging.getLogger(__name__)


class ProgressTracker:
    """
    Progress tracker with rich console output when available,
    fallback to simple logging when rich is not available.
    """
    
    def __init__(self, total: int, description: str = "Processing", 
                 disable: bool = False, quiet: bool = False):
        """
        Initialize progress tracker.
        
        Args:
            total: Total number of items to process
            description: Description of the operation
            disable: Whether to disable progress tracking
            quiet: Whether to suppress output
        """
        self.total = total
        self.description = description
        self.disable = disable or total <= 0
        self.quiet = quiet
        
        self.current = 0
        self.start_time = datetime.now()
        self.last_update = self.start_time
        
        # Rich progress components
        self.progress: Optional[Progress] = None
        self.task_id: Optional[TaskID] = None
        self.console: Optional[Console] = None
        
        # Fallback tracking
        self.last_logged_percent = -1
        
        if not self.disable and not self.quiet:
            self._setup_progress()
    
    def _setup_progress(self):
        """Set up progress tracking interface."""
        if HAS_RICH and not self.quiet:
            try:
                self.console = Console()
                self.progress = Progress(
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
                    TextColumn("({task.completed}/{task.total})"),
                    TimeRemainingColumn(),
                    console=self.console,
                    transient=True
                )
                self.progress.start()
                self.task_id = self.progress.add_task(
                    self.description, 
                    total=self.total
                )
                logger.debug(f"Rich progress tracker initialized: {self.description}")
            except Exception as e:
                logger.warning(f"Failed to initialize rich progress: {e}")
                self.progress = None
        else:
            logger.debug(f"Fallback progress tracker initialized: {self.description}")
    
    def update(self, increment: int = 1, description: Optional[str] = None):
        """
        Update progress.
        
        Args:
            increment: Number of items completed
            description: Optional new description
        """
        if self.disable:
            return
        
        self.current += increment
        self.last_update = datetime.now()
        
        # Update description if provided
        if description:
            self.description = description
        
        # Update rich progress if available
        if self.progress and self.task_id is not None:
            try:
                self.progress.update(
                    self.task_id, 
                    advance=increment,
                    description=self.description
                )
            except Exception as e:
                logger.warning(f"Failed to update rich progress: {e}")
                self._fallback_log_progress()
        else:
            self._fallback_log_progress()
    
    def _fallback_log_progress(self):
        """Log progress using standard logging (fallback)."""
        if self.quiet or self.total <= 0:
            return
        
        percent = (self.current / self.total) * 100
        
        # Only log at certain intervals to avoid spam
        percent_threshold = 10  # Log every 10%
        current_threshold = int(percent // percent_threshold) * percent_threshold
        
        if current_threshold > self.last_logged_percent:
            elapsed = self.elapsed_time
            if self.current > 0 and elapsed > 0:
                rate = self.current / elapsed
                remaining_items = self.total - self.current
                eta_seconds = remaining_items / rate if rate > 0 else 0
                eta = timedelta(seconds=int(eta_seconds))
                
                logger.info(
                    f"{self.description}: {self.current}/{self.total} "
                    f"({percent:.1f}%) - ETA: {eta}"
                )
            else:
                logger.info(
                    f"{self.description}: {self.current}/{self.total} "
                    f"({percent:.1f}%)"
                )
            
            self.last_logged_percent = current_threshold
    
    def set_description(self, description: str):
        """Update the progress description."""
        self.description = description
        
        if self.progress and self.task_id is not None:
            try:
                self.progress.update(self.task_id, description=description)
            except Exception:
                pass
    
    def close(self):
        """Close the progress tracker."""
        if self.progress:
            try:
                self.progress.stop()
            except Exception as e:
                logger.warning(f"Failed to stop rich progress: {e}")
        
        # Log final summary
        if not self.quiet and not self.disable:
            elapsed = self.elapsed_time
            rate = self.current / elapsed if elapsed > 0 else 0
            
            logger.info(
                f"{self.description} completed: {self.current}/{self.total} items "
                f"in {elapsed:.1f}s ({rate:.1f} items/sec)"
            )
    
    @property
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        return (datetime.now() - self.start_time).total_seconds()
    
    @property
    def completion_percentage(self) -> float:
        """Get completion percentage."""
        if self.total <= 0:
            return 0.0
        return (self.current / self.total) * 100
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class SimpleProgressTracker:
    """
    Simplified progress tracker for cases where rich is not needed.
    """
    
    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.description = description
        self.current = 0
        self.start_time = datetime.now()
    
    def update(self, increment: int = 1):
        """Update progress."""
        self.current += increment
    
    def close(self):
        """Close tracker."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        logger.info(f"{self.description} completed: {self.current} items in {elapsed:.1f}s")


def track_progress(iterable, description: str = "Processing", 
                  disable: bool = False, quiet: bool = False):
    """
    Track progress over an iterable.
    
    Args:
        iterable: Items to iterate over
        description: Description of the operation
        disable: Whether to disable progress tracking
        quiet: Whether to suppress output
        
    Yields:
        Items from the iterable
    """
    items = list(iterable)  # Convert to list to get length
    total = len(items)
    
    with ProgressTracker(total, description, disable, quiet) as tracker:
        for item in items:
            yield item
            tracker.update(1)


class BatchProgressTracker:
    """
    Progress tracker for batch operations with multiple stages.
    """
    
    def __init__(self, stages: list, quiet: bool = False):
        """
        Initialize batch progress tracker.
        
        Args:
            stages: List of stage names
            quiet: Whether to suppress output
        """
        self.stages = stages
        self.quiet = quiet
        self.current_stage = 0
        self.current_stage_progress = 0
        self.stage_totals = {}
        self.start_time = datetime.now()
        
        # Rich components
        self.console = Console() if HAS_RICH and not quiet else None
        self.progress = None
        self.overall_task = None
        self.stage_tasks = {}
        
        if self.console:
            self._setup_batch_progress()
    
    def _setup_batch_progress(self):
        """Set up batch progress interface."""
        if not HAS_RICH or self.quiet:
            return
        
        try:
            self.progress = Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=self.console
            )
            self.progress.start()
            
            # Add overall progress task
            self.overall_task = self.progress.add_task(
                "Overall Progress", 
                total=len(self.stages)
            )
            
            # Add tasks for each stage
            for stage in self.stages:
                task_id = self.progress.add_task(
                    f"  {stage}", 
                    total=100  # Will be updated when stage starts
                )
                self.stage_tasks[stage] = task_id
                
        except Exception as e:
            logger.warning(f"Failed to initialize batch progress: {e}")
            self.progress = None
    
    def start_stage(self, stage: str, total_items: int):
        """Start a new stage."""
        if stage not in self.stages:
            logger.warning(f"Unknown stage: {stage}")
            return
        
        self.current_stage = self.stages.index(stage)
        self.current_stage_progress = 0
        self.stage_totals[stage] = total_items
        
        if self.progress and stage in self.stage_tasks:
            self.progress.update(
                self.stage_tasks[stage], 
                total=total_items,
                completed=0
            )
        
        logger.info(f"Starting stage: {stage} ({total_items} items)")
    
    def update_stage(self, stage: str, increment: int = 1):
        """Update progress for current stage."""
        self.current_stage_progress += increment
        
        if self.progress and stage in self.stage_tasks:
            self.progress.update(
                self.stage_tasks[stage], 
                advance=increment
            )
    
    def complete_stage(self, stage: str):
        """Mark a stage as complete."""
        if self.progress and self.overall_task:
            self.progress.update(self.overall_task, advance=1)
        
        logger.info(f"Completed stage: {stage}")
    
    def close(self):
        """Close the batch progress tracker."""
        if self.progress:
            try:
                self.progress.stop()
            except Exception as e:
                logger.warning(f"Failed to stop batch progress: {e}")
        
        # Log summary
        elapsed = (datetime.now() - self.start_time).total_seconds()
        logger.info(f"Batch processing completed in {elapsed:.1f}s")


def create_progress_callback(tracker: ProgressTracker) -> Callable:
    """
    Create a callback function for use with other libraries.
    
    Args:
        tracker: Progress tracker instance
        
    Returns:
        Callback function that updates the tracker
    """
    def callback(increment: int = 1):
        tracker.update(increment)
    
    return callback


class PerformanceMonitor:
    """
    Monitor performance metrics during processing.
    """
    
    def __init__(self):
        self.start_time = None
        self.checkpoints = []
        self.memory_samples = []
    
    def start(self):
        """Start monitoring."""
        self.start_time = datetime.now()
        self.checkpoint("start")
    
    def checkpoint(self, name: str):
        """Add a performance checkpoint."""
        now = datetime.now()
        if self.start_time:
            elapsed = (now - self.start_time).total_seconds()
        else:
            elapsed = 0
        
        checkpoint_data = {
            'name': name,
            'timestamp': now,
            'elapsed': elapsed
        }
        
        # Try to get memory usage
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            checkpoint_data['memory_mb'] = memory_mb
        except ImportError:
            pass
        
        self.checkpoints.append(checkpoint_data)
        logger.debug(f"Performance checkpoint '{name}': {elapsed:.2f}s")
    
    def get_summary(self) -> dict:
        """Get performance summary."""
        if not self.checkpoints:
            return {}
        
        total_time = self.checkpoints[-1]['elapsed']
        
        summary = {
            'total_time': total_time,
            'checkpoints': self.checkpoints,
            'checkpoint_count': len(self.checkpoints)
        }
        
        # Add memory info if available
        memory_values = [cp.get('memory_mb') for cp in self.checkpoints if 'memory_mb' in cp]
        if memory_values:
            summary['peak_memory_mb'] = max(memory_values)
            summary['avg_memory_mb'] = sum(memory_values) / len(memory_values)
        
        return summary
