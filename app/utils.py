"""
Utilities Module

Contains utility functions used across the application.
"""

import os
import logging
from pathlib import Path
from typing import Optional
from loguru import logger
import sys

def setup_logger(name: Optional[str] = None) -> logger:
    """
    Set up a logger with proper formatting.
    
    Args:
        name: Optional name for the logger
        
    Returns:
        Configured logger instance
    """
    # Remove default handler
    logger.remove()
    
    # Add a handler with better formatting
    log_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    
    # Add console handler if debugging
    logger.add(sys.stderr, format=log_format, level="DEBUG" if os.getenv("DEBUG") else "INFO")
    
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Add file handler
    log_file = "app.log" if not name else f"{name}.log"
    logger.add(
        logs_dir / log_file,
        rotation="10 MB",
        retention="7 days",
        format=log_format,
        level="DEBUG",
        backtrace=True,
        diagnose=True
    )
    
    # Create a named logger if specified
    if name:
        return logger.bind(name=name)
    
    return logger

def create_directory_if_not_exists(directory_path: str) -> None:
    """
    Create a directory if it doesn't exist.
    
    Args:
        directory_path: Path to the directory to create
    """
    path = Path(directory_path)
    path.mkdir(parents=True, exist_ok=True)

def is_url(url: str) -> bool:
    """
    Check if a string is a valid URL.
    
    Args:
        url: String to check
        
    Returns:
        Whether the string is a valid URL
    """
    if not url:
        return False
    
    # Basic check for http/https
    return url.startswith(("http://", "https://", "ftp://"))

def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename to be safe for the filesystem.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Replace problematic characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Limit length
    if len(filename) > 255:
        filename = filename[:255]
    
    return filename

def get_file_extension(url: str) -> str:
    """
    Get file extension from URL.
    
    Args:
        url: URL to extract extension from
        
    Returns:
        File extension or empty string
    """
    if not url:
        return ""
    
    # Split URL by '?' to remove query parameters
    url_without_query = url.split('?')[0]
    
    # Get the filename from the URL
    filename = url_without_query.split('/')[-1]
    
    # Split by '.' and get the last part
    parts = filename.split('.')
    if len(parts) > 1:
        return parts[-1].lower()
    
    return ""

def is_image_url(url: str) -> bool:
    """
    Check if URL points to an image.
    
    Args:
        url: URL to check
        
    Returns:
        Whether the URL points to an image
    """
    image_extensions = ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp', 'svg']
    extension = get_file_extension(url)
    return extension in image_extensions

def truncate_text(text: str, max_length: int = 100) -> str:
    """
    Truncate text to a maximum length with ellipsis.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length-3] + "..."

def format_bytes(size: int) -> str:
    """
    Format bytes to human-readable size.
    
    Args:
        size: Size in bytes
        
    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    
    return f"{size:.2f} PB"