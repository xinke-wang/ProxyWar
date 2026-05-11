"""
Base Logger for ProxyWar framework
"""

import sys
import time
from typing import Optional, Any
from datetime import datetime
from enum import Enum


class LogLevel(Enum):
    """Log level enumeration"""
    DEBUG = 0
    INFO = 1
    SUCCESS = 2
    WARNING = 3
    ERROR = 4


class BaseLogger:
    """
    Base logger class for ProxyWar framework.
    
    Provides basic logging functionality that can be extended by specific loggers.
    """
    
    def __init__(self, name: str = "ProxyWar", log_level: LogLevel = LogLevel.INFO):
        """
        Initialize the base logger.
        
        Args:
            name: Name of the logger
            log_level: Minimum log level to display
        """
        self.name = name
        self.log_level = log_level
        self.start_time = time.time()
        
    def _should_log(self, level: LogLevel) -> bool:
        """Check if message should be logged based on current log level."""
        return level.value >= self.log_level.value
    
    def _format_message(self, level: LogLevel, message: str) -> str:
        """Format log message with timestamp and level."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        level_str = level.name
        return f"[{timestamp}] [{level_str}] {message}"
    
    def _write_message(self, formatted_message: str, level: LogLevel):
        """Write the formatted message to output."""
        if level == LogLevel.ERROR:
            print(formatted_message, file=sys.stderr)
        else:
            print(formatted_message)
    
    def log(self, level: LogLevel, message: str):
        """Log a message at the specified level."""
        if self._should_log(level):
            formatted_message = self._format_message(level, message)
            self._write_message(formatted_message, level)
    
    def debug(self, message: str):
        """Log a debug message."""
        self.log(LogLevel.DEBUG, message)
    
    def info(self, message: str):
        """Log an info message."""
        self.log(LogLevel.INFO, message)
    
    def success(self, message: str):
        """Log a success message."""
        self.log(LogLevel.SUCCESS, f"{message}")
    
    def warning(self, message: str):
        """Log a warning message."""
        self.log(LogLevel.WARNING, f" {message}")
    
    def error(self, message: str):
        """Log an error message."""
        self.log(LogLevel.ERROR, f"{message}")
    
    def failure(self, message: str):
        """Log a failure message."""
        self.log(LogLevel.ERROR, f"{message}")
    
    def section_header(self, title: str, width: int = 60):
        """Log a section header."""
        if title:
            border = "=" * width
            self.info(border)
            self.info(f" {title} ".center(width))
            self.info(border)
        else:
            self.info("=" * width)
    
    def subsection_header(self, title: str, width: int = 40):
        """Log a subsection header."""
        if title:
            border = "-" * width
            self.info(border)
            self.info(f" {title} ".center(width))
            self.info(border)
        else:
            self.info("-" * width) 