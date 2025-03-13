"""
File Manager

This module handles file operations for the application.
"""

import logging
import os
from pathlib import Path
import shutil
import json
import time
from typing import List, Dict, Any, Optional, BinaryIO
import asyncio
import aiofiles
import base64

logger = logging.getLogger(__name__)

async def save_to_file(content: str, filepath: str, mode: str = 'w') -> bool:
    """
    Save content to a file asynchronously.
    
    Args:
        content: Content to save
        filepath: Path to save the file
        mode: File open mode ('w' for text, 'wb' for binary)
        
    Returns:
        Boolean indicating success
    """
    try:
        # Create directory if it doesn't exist
        directory = os.path.dirname(filepath)
        os.makedirs(directory, exist_ok=True)
        
        # Save the file
        async with aiofiles.open(filepath, mode=mode) as f:
            if mode == 'wb' and isinstance(content, str):
                # If binary mode but string content, encode it
                await f.write(content.encode('utf-8'))
            else:
                await f.write(content)
        
        logger.info(f"Saved content to {filepath}")
        return True
    
    except Exception as e:
        logger.error(f"Error saving to file {filepath}: {str(e)}", exc_info=True)
        return False

async def read_from_file(filepath: str, mode: str = 'r') -> Optional[str]:
    """
    Read content from a file asynchronously.
    
    Args:
        filepath: Path to the file
        mode: File open mode ('r' for text, 'rb' for binary)
        
    Returns:
        File content or None if error
    """
    try:
        if not os.path.exists(filepath):
            logger.warning(f"File not found: {filepath}")
            return None
        
        # Read the file
        async with aiofiles.open(filepath, mode=mode) as f:
            content = await f.read()
            
            # If binary mode, decode to string
            if mode == 'rb' and isinstance(content, bytes):
                content = content.decode('utf-8')
        
        return content
    
    except Exception as e:
        logger.error(f"Error reading file {filepath}: {str(e)}", exc_info=True)
        return None

def save_base64_image(base64_str: str, output_dir: str, filename: Optional[str] = None) -> Optional[str]:
    """
    Save a base64-encoded image to a file.
    
    Args:
        base64_str: Base64-encoded image string
        output_dir: Directory to save the image
        filename: Optional filename, generated if not provided
        
    Returns:
        Path to the saved image or None if error
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Remove data URL prefix if present
        if ',' in base64_str:
            header, base64_str = base64_str.split(',', 1)
        
        # Decode base64
        image_data = base64.b64decode(base64_str)
        
        # Generate filename if not provided
        if not filename:
            timestamp = int(time.time())
            filename = f"image_{timestamp}.jpg"
        
        # Full path
        filepath = os.path.join(output_dir, filename)
        
        # Save the image
        with open(filepath, 'wb') as f:
            f.write(image_data)
        
        logger.info(f"Saved base64 image to {filepath}")
        return filepath
    
    except Exception as e:
        logger.error(f"Error saving base64 image: {str(e)}", exc_info=True)
        return None

async def list_directory(directory: str, pattern: str = "*") -> List[Dict[str, Any]]:
    """
    List files in a directory with metadata.
    
    Args:
        directory: Directory path
        pattern: Glob pattern for filtering files
        
    Returns:
        List of file information dictionaries
    """
    try:
        # Create Path object
        path = Path(directory)
        
        # Check if directory exists
        if not path.exists() or not path.is_dir():
            logger.warning(f"Directory not found: {directory}")
            return []
        
        # List files matching pattern
        files = list(path.glob(pattern))
        
        # Get file metadata
        file_info = []
        for file_path in files:
            if file_path.is_file():
                info = {
                    "name": file_path.name,
                    "path": str(file_path),
                    "size": file_path.stat().st_size,
                    "modified": file_path.stat().st_mtime,
                    "created": file_path.stat().st_ctime,
                    "is_file": True
                }
                file_info.append(info)
            elif file_path.is_dir():
                info = {
                    "name": file_path.name,
                    "path": str(file_path),
                    "modified": file_path.stat().st_mtime,
                    "created": file_path.stat().st_ctime,
                    "is_file": False
                }
                file_info.append(info)
        
        # Sort by modified time (newest first)
        file_info.sort(key=lambda x: x["modified"], reverse=True)
        
        return file_info
    
    except Exception as e:
        logger.error(f"Error listing directory {directory}: {str(e)}", exc_info=True)
        return []

async def json_to_markdown(json_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Convert a JSON file to Markdown files.
    
    Args:
        json_path: Path to the JSON file
        output_dir: Directory to save Markdown files (default: auto-generated)
        
    Returns:
        Dict with conversion results
    """
    try:
        logger.info(f"Converting JSON to Markdown: {json_path}")
        
        # Read JSON file
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract basic info
        base_url = data.get("base_url", "unknown")
        domain = data.get("domain", "unknown")
        timestamp = data.get("timestamp", int(time.time()))
        pages = data.get("pages", [])
        
        if not pages:
            logger.warning(f"No pages found in JSON file: {json_path}")
            return {
                "success": False,
                "error": "No pages found in JSON file",
                "json_path": json_path
            }
        
        # Create output directory if not provided
        if not output_dir:
            output_dir = os.path.join("data/markdown", f"{domain}_{timestamp}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert each page to Markdown
        markdown_files = []
        for page in pages:
            try:
                # Get page info
                url = page.get("url", "")
                title = page.get("title", "Untitled")
                
                # Clean URL for filename
                parsed_url = url.replace("://", "_").replace("/", "_").replace(".", "_")
                if len(parsed_url) > 100:
                    parsed_url = parsed_url[:100]
                
                filename = f"{parsed_url}.md"
                filepath = os.path.join(output_dir, filename)
                
                # Build Markdown content
                markdown_parts = [f"# {title}", f"Source: [{url}]({url})", ""]
                
                # Add metadata if available
                metadata = page.get("metadata", {})
                description = metadata.get("description", "")
                if description:
                    markdown_parts.append(f"> {description}")
                    markdown_parts.append("")
                
                # Add sections
                sections = page.get("content", {}).get("sections", [])
                for section in sections:
                    section_title = section.get("title", "")
                    section_level = section.get("level", 2)
                    section_content = section.get("content", "")
                    
                    if section_title and section_title != title:
                        heading_prefix = '#' * min(section_level, 6)
                        markdown_parts.append(f"{heading_prefix} {section_title}")
                        markdown_parts.append("")
                    
                    if section_content:
                        markdown_parts.append(section_content)
                        markdown_parts.append("")
                
                # Join and save
                markdown_content = "\n".join(markdown_parts)
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(markdown_content)
                
                markdown_files.append({
                    "title": title,
                    "path": filepath,
                    "url": url
                })
            
            except Exception as e:
                logger.warning(f"Error converting page to Markdown: {str(e)}")
        
        logger.info(f"Converted {len(markdown_files)} pages to Markdown in {output_dir}")
        
        return {
            "success": True,
            "output_dir": output_dir,
            "file_count": len(markdown_files),
            "files": markdown_files
        }
    
    except Exception as e:
        logger.error(f"Error converting JSON to Markdown: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "json_path": json_path
        }

def get_file_extension(filename: str) -> str:
    """
    Get the extension of a file.
    
    Args:
        filename: Name of the file
        
    Returns:
        File extension (lowercase, without dot)
    """
    _, ext = os.path.splitext(filename)
    return ext.lower().lstrip('.')

def get_mime_type(filename: str) -> str:
    """
    Get the MIME type of a file based on its extension.
    
    Args:
        filename: Name of the file
        
    Returns:
        MIME type
    """
    import mimetypes
    mime_type, _ = mimetypes.guess_type(filename)
    return mime_type or "application/octet-stream"

def create_download_link(content: str, filename: str, mime_type: Optional[str] = None) -> str:
    """
    Create a data URL for downloading content.
    
    Args:
        content: Content to download
        filename: Name of the file
        mime_type: MIME type of the content
        
    Returns:
        Data URL
    """
    try:
        # Determine MIME type if not provided
        if not mime_type:
            mime_type = get_mime_type(filename)
        
        # Encode content
        if isinstance(content, str):
            encoded_content = base64.b64encode(content.encode('utf-8')).decode('utf-8')
        else:
            encoded_content = base64.b64encode(content).decode('utf-8')
        
        # Create data URL
        data_url = f"data:{mime_type};base64,{encoded_content}"
        
        return data_url
    
    except Exception as e:
        logger.error(f"Error creating download link: {str(e)}", exc_info=True)
        return ""