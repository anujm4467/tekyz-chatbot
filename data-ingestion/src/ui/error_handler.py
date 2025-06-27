"""
Error Handler for Data Ingestion Pipeline UI
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import threading


@dataclass
class ErrorRecord:
    """Simple error record"""
    timestamp: datetime
    error_id: str
    step: str
    message: str
    details: str
    retry_count: int = 0
    resolved: bool = False


class ErrorHandler:
    """Simple error handler for Streamlit UI"""
    
    def __init__(self):
        self.errors: List[ErrorRecord] = []
        self.lock = threading.Lock()
    
    def add_error(self, message: str, step: str, details: str = "") -> str:
        """Add a new error"""
        with self.lock:
            error_id = f"error_{len(self.errors)}"
            error = ErrorRecord(
                timestamp=datetime.now(),
                error_id=error_id,
                step=step,
                message=message,
                details=details
            )
            self.errors.append(error)
            return error_id
    
    def get_recent_errors(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent errors"""
        with self.lock:
            recent = self.errors[-count:] if self.errors else []
            return [
                {
                    'timestamp': error.timestamp.strftime('%H:%M:%S'),
                    'step': error.step,
                    'message': error.message,
                    'details': error.details,
                    'retry_count': error.retry_count
                }
                for error in recent
            ] 