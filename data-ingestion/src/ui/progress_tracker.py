"""
Progress Tracker for Data Ingestion Pipeline

Provides real-time progress tracking, ETA estimation, and status monitoring
for the Streamlit UI.
"""

import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from collections import deque
import threading
import json
from pathlib import Path


@dataclass
class StepProgress:
    """Progress information for a single pipeline step"""
    name: str
    status: str = "pending"  # pending, running, completed, error
    progress: float = 0.0  # 0-100
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    items_processed: int = 0
    total_items: int = 0
    
    @property
    def duration(self) -> Optional[timedelta]:
        """Get the duration of this step"""
        if self.start_time:
            end = self.end_time or datetime.now()
            return end - self.start_time
        return None
    
    @property
    def items_per_second(self) -> float:
        """Get processing rate in items per second"""
        if self.duration and self.duration.total_seconds() > 0:
            return self.items_processed / self.duration.total_seconds()
        return 0.0


@dataclass
class PipelineMetrics:
    """Overall pipeline metrics and performance data"""
    total_items: int = 0
    processed_items: int = 0
    failed_items: int = 0
    start_time: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None
    current_step: Optional[str] = None
    overall_progress: float = 0.0
    
    @property
    def total_time_seconds(self) -> float:
        """Get total processing time in seconds"""
        if self.start_time:
            end_time = datetime.now()
            return (end_time - self.start_time).total_seconds()
        return 0.0
    
    # Historical data for ETA estimation
    historical_rates: Dict[str, List[float]] = field(default_factory=dict)
    step_weights: Dict[str, float] = field(default_factory=lambda: {
        'processing': 0.15,
        'chunking': 0.20,
        'embedding': 0.45,
        'upload': 0.15,
        'validation': 0.05
    })


class ProgressTracker:
    """
    Real-time progress tracking for the data ingestion pipeline
    
    Features:
    - Step-by-step progress monitoring
    - ETA estimation based on historical data
    - Real-time log streaming
    - Performance metrics collection
    """
    
    def __init__(self, max_log_entries: int = 1000):
        """
        Initialize the progress tracker
        
        Args:
            max_log_entries: Maximum number of log entries to keep in memory
        """
        self.steps: Dict[str, StepProgress] = {}
        self.metrics = PipelineMetrics()
        self.logs: deque = deque(maxlen=max_log_entries)
        self.is_running = False
        self.lock = threading.Lock()
        
        # Callbacks for real-time updates
        self.update_callbacks: List[Callable] = []
        
        # Initialize default steps
        self._initialize_steps()
    
    def _initialize_steps(self):
        """Initialize default pipeline steps"""
        default_steps = [
            "processing",
            "chunking", 
            "embedding",
            "upload",
            "validation"
        ]
        
        for step in default_steps:
            self.steps[step] = StepProgress(name=step)
    
    def start_pipeline(self, total_items: int = 0):
        """
        Start tracking a new pipeline run
        
        Args:
            total_items: Total number of items to process
        """
        with self.lock:
            self.metrics.total_items = total_items
            self.metrics.processed_items = 0
            self.metrics.failed_items = 0
            self.metrics.start_time = datetime.now()
            self.metrics.current_step = None
            self.metrics.overall_progress = 0.0
            self.is_running = True
            
            # Reset all steps
            for step in self.steps.values():
                step.status = "pending"
                step.progress = 0.0
                step.start_time = None
                step.end_time = None
                step.error_message = None
                step.items_processed = 0
                step.total_items = 0
            
            self.add_log("Pipeline started", "info")
    
    def start_step(self, step_name: str, total_items: int = 0):
        """
        Start a specific pipeline step
        
        Args:
            step_name: Name of the step to start
            total_items: Total items for this step
        """
        with self.lock:
            if step_name not in self.steps:
                self.steps[step_name] = StepProgress(name=step_name)
            
            step = self.steps[step_name]
            step.status = "running"
            step.start_time = datetime.now()
            step.total_items = total_items
            step.items_processed = 0
            step.progress = 0.0
            step.error_message = None
            
            self.metrics.current_step = step_name
            
            self.add_log(f"Started step: {step_name}", "info")
            self._update_overall_progress()
    
    def update_step_progress(self, step_name: str, progress: float, items_processed: int = None):
        """
        Update progress for a specific step
        
        Args:
            step_name: Name of the step to update
            progress: Progress percentage (0-100)
            items_processed: Number of items processed (optional)
        """
        with self.lock:
            if step_name not in self.steps:
                self.steps[step_name] = StepProgress(name=step_name)
            
            step = self.steps[step_name]
            step.progress = min(100.0, max(0.0, progress))
            
            if items_processed is not None:
                step.items_processed = items_processed
            
            self._update_overall_progress()
            self._update_eta_estimation()
    
    def update_step(self, step_name: str, progress: float, message: str = None):
        """
        Update progress for a specific step with optional message
        
        Args:
            step_name: Name of the step to update
            progress: Progress percentage (0-100)
            message: Optional status message
        """
        with self.lock:
            if step_name not in self.steps:
                self.steps[step_name] = StepProgress(name=step_name)
            
            step = self.steps[step_name]
            if step.status != "running":
                step.status = "running"
                step.start_time = datetime.now()
                self.metrics.current_step = step_name
            
            step.progress = min(100.0, max(0.0, progress))
            
            if message:
                self.add_log(f"{step_name}: {message}", "info")
            
            self._update_overall_progress()
            self._update_eta_estimation()
    
    def complete_step(self, step_name: str, success: bool = True, error_message: str = None):
        """
        Mark a step as completed
        
        Args:
            step_name: Name of the step to complete
            success: Whether the step completed successfully
            error_message: Error message if step failed
        """
        with self.lock:
            if step_name not in self.steps:
                self.steps[step_name] = StepProgress(name=step_name)
            
            step = self.steps[step_name]
            step.end_time = datetime.now()
            step.status = "completed" if success else "error"
            step.progress = 100.0 if success else step.progress
            step.error_message = error_message
            
            # Record performance metrics for ETA estimation
            if success and step.duration:
                if step_name not in self.metrics.historical_rates:
                    self.metrics.historical_rates[step_name] = []
                
                rate = step.items_per_second
                self.metrics.historical_rates[step_name].append(rate)
                
                # Keep only recent rates (last 10 runs)
                if len(self.metrics.historical_rates[step_name]) > 10:
                    self.metrics.historical_rates[step_name].pop(0)
            
            log_level = "info" if success else "error"
            log_message = f"Completed step: {step_name}"
            if error_message:
                log_message += f" - Error: {error_message}"
                
            self.add_log(log_message, log_level)
            self._update_overall_progress()
    
    def add_log(self, message: str, level: str = "info"):
        """
        Add a log entry
        
        Args:
            message: Log message
            level: Log level (info, warning, error)
        """
        with self.lock:
            timestamp = datetime.now().strftime("%H:%M:%S")
            log_entry = f"[{timestamp}] {level.upper()}: {message}"
            self.logs.append(log_entry)
            
            # Trigger callbacks for real-time updates
            for callback in self.update_callbacks:
                try:
                    callback()
                except Exception:
                    pass  # Ignore callback errors
    
    def get_overall_progress(self) -> float:
        """Get overall pipeline progress percentage"""
        with self.lock:
            return self.metrics.overall_progress
    
    def get_step_status(self, step_name: str) -> str:
        """Get status of a specific step"""
        with self.lock:
            return self.steps.get(step_name, StepProgress(name=step_name)).status
    
    def get_step_progress(self, step_name: str) -> float:
        """Get progress of a specific step"""
        with self.lock:
            return self.steps.get(step_name, StepProgress(name=step_name)).progress
    
    def get_estimated_completion_time(self) -> Optional[datetime]:
        """Get estimated completion time"""
        with self.lock:
            return self.metrics.estimated_completion
    
    def get_recent_logs(self, count: int = 20) -> List[str]:
        """Get recent log entries"""
        with self.lock:
            return list(self.logs)[-count:]
    
    def get_pipeline_metrics(self) -> Dict[str, Any]:
        """Get comprehensive pipeline metrics"""
        with self.lock:
            metrics = {
                'total_items': self.metrics.total_items,
                'processed_items': self.metrics.processed_items,
                'failed_items': self.metrics.failed_items,
                'overall_progress': self.metrics.overall_progress,
                'current_step': self.metrics.current_step,
                'is_running': self.is_running,
                'start_time': self.metrics.start_time.isoformat() if self.metrics.start_time else None,
                'estimated_completion': self.metrics.estimated_completion.isoformat() if self.metrics.estimated_completion else None,
                'steps': {}
            }
            
            for name, step in self.steps.items():
                metrics['steps'][name] = {
                    'status': step.status,
                    'progress': step.progress,
                    'items_processed': step.items_processed,
                    'total_items': step.total_items,
                    'duration': step.duration.total_seconds() if step.duration else None,
                    'items_per_second': step.items_per_second,
                    'error_message': step.error_message
                }
            
            return metrics
    
    def stop_pipeline(self, success: bool = False):
        """
        Stop the pipeline tracking
        
        Args:
            success: Whether the pipeline completed successfully
        """
        with self.lock:
            self.is_running = False
            
            if not success:
                # Mark current step as failed if pipeline was stopped
                if self.metrics.current_step:
                    self.complete_step(
                        self.metrics.current_step, 
                        success=False, 
                        error_message="Pipeline stopped by user"
                    )
            
            self.add_log(f"Pipeline {'completed' if success else 'stopped'}", "info")
    
    def _update_overall_progress(self):
        """Update overall pipeline progress based on step weights"""
        total_weighted_progress = 0.0
        total_weight = 0.0
        
        for step_name, step in self.steps.items():
            weight = self.metrics.step_weights.get(step_name, 0.1)
            total_weighted_progress += step.progress * weight
            total_weight += weight
        
        if total_weight > 0:
            self.metrics.overall_progress = total_weighted_progress / total_weight
        else:
            self.metrics.overall_progress = 0.0
    
    def _update_eta_estimation(self):
        """Update ETA estimation based on current progress and historical data"""
        if not self.metrics.start_time or self.metrics.overall_progress <= 0:
            return
        
        current_time = datetime.now()
        elapsed_time = current_time - self.metrics.start_time
        
        # Simple linear projection based on overall progress
        if self.metrics.overall_progress > 0:
            total_estimated_time = elapsed_time * (100.0 / self.metrics.overall_progress)
            remaining_time = total_estimated_time - elapsed_time
            self.metrics.estimated_completion = current_time + remaining_time
        
        # More sophisticated estimation using historical rates (if available)
        # This could be enhanced with machine learning models
    
    def add_update_callback(self, callback: Callable):
        """Add a callback for real-time updates"""
        self.update_callbacks.append(callback)
    
    def remove_update_callback(self, callback: Callable):
        """Remove an update callback"""
        if callback in self.update_callbacks:
            self.update_callbacks.remove(callback)
    
    def export_metrics(self, file_path: str):
        """
        Export metrics to a JSON file
        
        Args:
            file_path: Path to save the metrics file
        """
        metrics = self.get_pipeline_metrics()
        
        with open(file_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
    
    def load_historical_data(self, file_path: str):
        """
        Load historical performance data for better ETA estimation
        
        Args:
            file_path: Path to historical data file
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            if 'historical_rates' in data:
                self.metrics.historical_rates.update(data['historical_rates'])
                
        except (FileNotFoundError, json.JSONDecodeError):
            pass  # Ignore errors, will build historical data over time 