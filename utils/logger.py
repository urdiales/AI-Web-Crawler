"""
Logger

This module sets up logging for the application.
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
import time

def setup_logger(log_file=None, log_level=logging.INFO):
    """
    Set up application logging.
    
    Args:
        log_file: Path to log file (optional)
        log_level: Logging level
    """
    # Create logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    
    # Add console handler to logger
    root_logger.addHandler(console_handler)
    
    # If log file provided, add file handler
    if log_file:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        # Create rotating file handler (10MB max, 5 backup files)
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        
        # Add file handler to logger
        root_logger.addHandler(file_handler)
    
    # Log setup complete
    root_logger.info("Logging initialized")

def get_logger(name):
    """
    Get a logger for a specific module.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)

class StreamToLogger:
    """
    Stream-like object that redirects writes to a logger instance.
    """
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''
    
    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())
    
    def flush(self):
        pass

def redirect_stdout_stderr_to_logger():
    """
    Redirect stdout and stderr to logger.
    """
    stdout_logger = logging.getLogger('STDOUT')
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl
    
    stderr_logger = logging.getLogger('STDERR')
    sl = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl
    
    logging.info("Redirected stdout and stderr to logger")

def log_execution_time(func):
    """
    Decorator to log the execution time of a function.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        start_time = time.time()
        logger.debug(f"Starting {func.__name__}")
        
        result = func(*args, **kwargs)
        
        elapsed_time = time.time() - start_time
        logger.debug(f"Completed {func.__name__} in {elapsed_time:.3f} seconds")
        
        return result
    
    return wrapper

def log_async_execution_time(func):
    """
    Decorator to log the execution time of an async function.
    
    Args:
        func: Async function to decorate
        
    Returns:
        Decorated async function
    """
    import functools
    
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        start_time = time.time()
        logger.debug(f"Starting async {func.__name__}")
        
        result = await func(*args, **kwargs)
        
        elapsed_time = time.time() - start_time
        logger.debug(f"Completed async {func.__name__} in {elapsed_time:.3f} seconds")
        
        return result
    
    return wrapper