"""
Error Handler

This module provides centralized error handling functionality for the application.
"""

import logging
import traceback
import sys
from typing import Dict, Any, Optional, Callable, Type
import functools
import asyncio
import time
import aiohttp
import requests
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Global registry of error handlers
error_handlers = {}

def setup_error_handlers():
    """Set up the application's error handlers."""
    # Register built-in error handlers
    register_error_handler(aiohttp.ClientError, handle_network_error)
    register_error_handler(requests.exceptions.RequestException, handle_network_error)
    register_error_handler(ValueError, handle_value_error)
    register_error_handler(TimeoutError, handle_timeout_error)
    register_error_handler(asyncio.TimeoutError, handle_timeout_error)
    register_error_handler(KeyError, handle_key_error)
    register_error_handler(FileNotFoundError, handle_file_error)
    register_error_handler(PermissionError, handle_file_error)
    
    logger.info("Error handlers registered")

def register_error_handler(error_type: Type[Exception], handler: Callable):
    """
    Register an error handler for a specific exception type.
    
    Args:
        error_type: Type of exception to handle
        handler: Handler function
    """
    error_handlers[error_type] = handler
    logger.debug(f"Registered error handler for {error_type.__name__}")

def handle_error(exc: Exception) -> Dict[str, Any]:
    """
    Handle an exception using the appropriate registered handler.
    
    Args:
        exc: Exception to handle
        
    Returns:
        Dictionary with error information
    """
    # Find the most specific handler for this exception type
    handler = None
    for error_type, error_handler in error_handlers.items():
        if isinstance(exc, error_type):
            handler = error_handler
            break
    
    # If no specific handler, use generic handler
    if handler is None:
        return handle_generic_error(exc)
    
    # Call the handler
    return handler(exc)

def handle_generic_error(exc: Exception) -> Dict[str, Any]:
    """
    Generic error handler for unhandled exceptions.
    
    Args:
        exc: Exception to handle
        
    Returns:
        Dictionary with error information
    """
    logger.error(f"Unhandled error: {str(exc)}", exc_info=True)
    
    return {
        "success": False,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback": traceback.format_exc(),
        "timestamp": time.time()
    }

def handle_network_error(exc: Exception) -> Dict[str, Any]:
    """
    Handle network-related errors.
    
    Args:
        exc: Network exception
        
    Returns:
        Dictionary with error information
    """
    logger.error(f"Network error: {str(exc)}", exc_info=True)
    
    # Extract URL if available
    url = None
    if hasattr(exc, 'request') and hasattr(exc.request, 'url'):
        url = exc.request.url
    
    return {
        "success": False,
        "error_type": "NetworkError",
        "error_message": str(exc),
        "url": url,
        "suggestion": "Please check your network connection and try again.",
        "timestamp": time.time()
    }

def handle_invalid_url(url: str) -> Dict[str, Any]:
    """
    Handle invalid URL errors.
    
    Args:
        url: Invalid URL
        
    Returns:
        Dictionary with error information
    """
    logger.error(f"Invalid URL: {url}")
    
    return {
        "success": False,
        "error_type": "InvalidURL",
        "error_message": f"The URL '{url}' is not valid",
        "url": url,
        "suggestion": "Please check the URL and try again.",
        "timestamp": time.time()
    }

def handle_timeout_error(exc: Exception) -> Dict[str, Any]:
    """
    Handle timeout errors.
    
    Args:
        exc: Timeout exception
        
    Returns:
        Dictionary with error information
    """
    logger.error(f"Timeout error: {str(exc)}", exc_info=True)
    
    return {
        "success": False,
        "error_type": "TimeoutError",
        "error_message": "The operation timed out",
        "details": str(exc),
        "suggestion": "The server took too long to respond. Please try again later.",
        "timestamp": time.time()
    }

def handle_rate_limit_error(status_code: int, response_text: str) -> Dict[str, Any]:
    """
    Handle rate limit errors.
    
    Args:
        status_code: HTTP status code
        response_text: Response text
        
    Returns:
        Dictionary with error information
    """
    logger.error(f"Rate limit error: {status_code} - {response_text}")
    
    retry_after = None
    if "retry-after" in response_text.lower():
        # Try to extract retry-after value
        try:
            retry_after = int(response_text.split("retry-after:")[1].split("\n")[0].strip())
        except Exception:
            pass
    
    return {
        "success": False,
        "error_type": "RateLimitError",
        "error_message": "Rate limit exceeded",
        "status_code": status_code,
        "details": response_text,
        "retry_after": retry_after,
        "suggestion": "You've reached the rate limit. Please wait and try again later.",
        "timestamp": time.time()
    }

def handle_value_error(exc: ValueError) -> Dict[str, Any]:
    """
    Handle value errors.
    
    Args:
        exc: ValueError exception
        
    Returns:
        Dictionary with error information
    """
    logger.error(f"Value error: {str(exc)}", exc_info=True)
    
    return {
        "success": False,
        "error_type": "ValueError",
        "error_message": str(exc),
        "suggestion": "The provided value is invalid. Please check your input and try again.",
        "timestamp": time.time()
    }

def handle_key_error(exc: KeyError) -> Dict[str, Any]:
    """
    Handle key errors.
    
    Args:
        exc: KeyError exception
        
    Returns:
        Dictionary with error information
    """
    logger.error(f"Key error: {str(exc)}", exc_info=True)
    
    return {
        "success": False,
        "error_type": "KeyError",
        "error_message": f"Missing required key: {str(exc)}",
        "suggestion": "A required value is missing. Please check your input and try again.",
        "timestamp": time.time()
    }

def handle_file_error(exc: Exception) -> Dict[str, Any]:
    """
    Handle file-related errors.
    
    Args:
        exc: File exception
        
    Returns:
        Dictionary with error information
    """
    logger.error(f"File error: {str(exc)}", exc_info=True)
    
    error_type = type(exc).__name__
    suggestion = ""
    
    if isinstance(exc, FileNotFoundError):
        suggestion = "The specified file could not be found. Please check the file path and try again."
    elif isinstance(exc, PermissionError):
        suggestion = "You don't have permission to access this file. Please check file permissions."
    else:
        suggestion = "An error occurred while accessing the file. Please check the file and try again."
    
    return {
        "success": False,
        "error_type": error_type,
        "error_message": str(exc),
        "suggestion": suggestion,
        "timestamp": time.time()
    }