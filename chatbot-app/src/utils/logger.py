"""
Logging Utility

This module provides centralized logging setup for the Tekyz chatbot application.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from loguru import logger as loguru_logger


def setup_logger(
    name: str = "tekyz_chatbot",
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: str = None
) -> logging.Logger:
    """
    Set up a logger with both console and file output using loguru.
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        format_string: Optional custom format string
    
    Returns:
        Configured logger instance
    """
    # Remove default loguru logger
    loguru_logger.remove()
    
    # Default format
    if format_string is None:
        format_string = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
    
    # Add console handler
    loguru_logger.add(
        sys.stdout,
        format=format_string,
        level=level,
        colorize=True
    )
    
    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        loguru_logger.add(
            log_file,
            format=format_string,
            level=level,
            rotation="10 MB",
            retention="7 days",
            compression="zip"
        )
    
    # Create a standard logger that forwards to loguru
    class InterceptHandler(logging.Handler):
        def emit(self, record):
            # Get corresponding Loguru level if it exists
            try:
                level = loguru_logger.level(record.levelname).name
            except ValueError:
                level = record.levelno

            # Find caller from where originated the logged message
            frame, depth = logging.currentframe(), 2
            while frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1

            loguru_logger.opt(depth=depth, exception=record.exc_info).log(
                level, record.getMessage()
            )

    # Set up standard logging to forward to loguru
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    
    return logging.getLogger(name)


class ChatbotLogger:
    """Specialized logger for chatbot operations."""
    
    def __init__(self, name: str = "tekyz_chatbot"):
        self.logger = setup_logger(name)
    
    def log_query(self, user_query: str, session_id: str):
        """Log user query."""
        self.logger.info(f"Query received [Session: {session_id}]: {user_query[:100]}...")
    
    def log_response(self, response: str, session_id: str, processing_time: float):
        """Log bot response."""
        self.logger.info(
            f"Response generated [Session: {session_id}] in {processing_time:.2f}s: "
            f"{response[:100]}..."
        )
    
    def log_error(self, error: Exception, context: str = ""):
        """Log error with context."""
        self.logger.error(f"Error in {context}: {str(error)}", exc_info=True)
    
    def log_search_results(self, query: str, results_count: int, processing_time: float):
        """Log search results."""
        self.logger.info(
            f"Vector search completed for '{query[:50]}...': "
            f"{results_count} results in {processing_time:.2f}s"
        )
    
    def log_classification(self, query: str, is_tekyz_related: bool, confidence: float):
        """Log intent classification results."""
        self.logger.info(
            f"Query classified: '{query[:50]}...' -> "
            f"Tekyz-related: {is_tekyz_related} (confidence: {confidence:.2f})"
        )
    
    def log_performance(self, operation: str, duration: float, details: dict = None):
        """Log performance metrics."""
        details_str = f" | Details: {details}" if details else ""
        self.logger.info(f"Performance | {operation}: {duration:.2f}s{details_str}")


# Global logger instance
_logger_instance: Optional[ChatbotLogger] = None


def get_logger() -> ChatbotLogger:
    """Get the global logger instance."""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = ChatbotLogger()
    return _logger_instance 