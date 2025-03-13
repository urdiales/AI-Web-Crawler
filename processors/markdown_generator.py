"""
Markdown Generator

This module handles converting processed content into clean Markdown
optimized for RAG systems.
"""

import logging
from typing import Dict, Any, List
import re
from urllib.parse import urlparse, urljoin

logger = logging.getLogger(__name__)

def generate_markdown(page_data: Dict[str, Any]) -> str:
    """
    Generate clean Markdown from processed page data.
    
    Args:
        page_data: Dictionary with processed page information
        
    Returns:
        Markdown string
    """
    try:
        # Truncate if too long
        if len(path) > 100:
            path = path[:100]
    
        # Create base filename
        base_filename = f"{parsed_url.netloc}__{path}"
    
        # Create list of filenames (one per chunk)
        filenames = [f"{base_path}/{base_filename}.md"]
    
        # If Crawl4AI already gave us markdown, use it
        if 'markdown' in page_data and page_data['markdown']:
            # Enhance the existing markdown
            enhanced_markdown = enhance_existing_markdown(page_data)
            return enhanced_markdown
        
        # Otherwise build from sections
        url = page_data.get('url', '')
        title = page_data.get('title', 'Untitled Page')
        metadata = page_data.get('metadata', {})
        sections = page_data.get('sections', [])
        
        markdown_parts = []
        
        # Add title
        markdown_parts.append(f"# {title}\n")
        
        # Add metadata
        markdown_parts.append(f"Source: [{url}]({url})\n")
        
        # Add description if available
        description = metadata.get('description', '')
        if description:
            markdown_parts.append(f"> {description}\n")
        
        # Add sections
        for section in sections:
            section_title = section.get('title', '')
            section_level = section.get('level', 2)
            section_content = section.get('content', '')
            
            # Skip empty sections
            if not section_content.strip():
                continue
            
            # Add section heading (if not the same as page title)
            if section_title and section_title != title:
                heading_prefix = '#' * min(section_level, 6)
                markdown_parts.append(f"{heading_prefix} {section_title}\n")
            
            # Add content with basic formatting
            cleaned_content = clean_text_for_markdown(section_content)
            markdown_parts.append(f"{cleaned_content}\n")
        
        # Join all parts
        markdown = '\n'.join(markdown_parts)
        
        return markdown
    
    except Exception as e:
        logger.error(f"Error generating markdown: {str(e)}", exc_info=True)
        return f"# {page_data.get('title', 'Error')}\n\nError generating markdown: {str(e)}"

def enhance_existing_markdown(page_data: Dict[str, Any]) -> str:
    """
    Enhance existing markdown with metadata and structure improvements.
    
    Args:
        page_data: Dictionary with page information including markdown
        
    Returns:
        Enhanced markdown string
    """
    try:
        markdown = page_data.get('markdown', '')
        url = page_data.get('url', '')
        title = page_data.get('title', 'Untitled Page')
        metadata = page_data.get('metadata', {})
        
        # Check if the markdown already contains the title
        has_title = re.search(r'^#\s+.+', markdown, re.MULTILINE) is not None
        
        markdown_parts = []
        
        # Add title if needed
        if not has_title:
            markdown_parts.append(f"# {title}\n")
        
        # Add metadata if not already present
        if 'Source:' not in markdown:
            markdown_parts.append(f"Source: [{url}]({url})\n")
        
        # Add description if available and not already present
        description = metadata.get('description', '')
        if description and '>' not in markdown[:200]:  # Quick check in first 200 chars
            markdown_parts.append(f"> {description}\n")
        
        # Add the original markdown
        markdown_parts.append(markdown)
        
        # Join all parts
        enhanced_markdown = '\n'.join(markdown_parts)
        
        # Fix common markdown issues
        enhanced_markdown = fix_markdown_formatting(enhanced_markdown)
        
        return enhanced_markdown
    
    except Exception as e:
        logger.error(f"Error enhancing markdown: {str(e)}", exc_info=True)
        return page_data.get('markdown', '')

def clean_text_for_markdown(text: str) -> str:
    """
    Clean text content for markdown formatting.
    
    Args:
        text: Text content
        
    Returns:
        Cleaned text suitable for markdown
    """
    # Normalize whitespace
    cleaned = re.sub(r'\s+', ' ', text).strip()
    
    # Escape markdown special characters (except in code blocks)
    # This is a simplified approach; a full solution would parse blocks
    special_chars = ['\\', '`', '*', '_', '{', '}', '[', ']', '(', ')', '#', '+', '-', '.', '!']
    
    for char in special_chars:
        # Skip if inside code blocks
        if char != '`':  # Don't escape backticks themselves
            cleaned = cleaned.replace(char, '\\' + char)
    
    return cleaned

def fix_markdown_formatting(markdown: str) -> str:
    """
    Fix common markdown formatting issues.
    
    Args:
        markdown: Markdown content
        
    Returns:
        Fixed markdown
    """
    # Ensure headings have space after #
    markdown = re.sub(r'(^|\n)#+([^#\s])', r'\1# \2', markdown)
    
    # Fix consecutive newlines (more than 2)
    markdown = re.sub(r'\n{3,}', '\n\n', markdown)
    
    # Ensure code blocks have proper spacing
    markdown = re.sub(r'```(\w+)(?!\n)', r'```\1\n', markdown)
    markdown = re.sub(r'(?<!\n)```', r'\n```', markdown)
    
    # Fix broken links
    markdown = re.sub(r'\[([^\]]+)\]\s+\(([^)]+)\)', r'[\1](\2)', markdown)
    
    return markdown

def create_chunk_filenames(page_data: Dict[str, Any], base_path: str) -> List[str]:
    """
    Create filenames for chunked markdown files.
    
    Args:
        page_data: Dictionary with page information
        base_path: Base directory path for files
        
    Returns:
        List of filenames
    """
    url = page_data.get('url', '')
    
    # Create a filename-safe version of the URL
    parsed_url = urlparse(url)
    path = parsed_url.path
    
    if not path or path == '/':
        path = 'index'
    else:
        # Remove leading/trailing slashes and replace invalid characters
        path = path.strip('/')
        path = re.sub(r'[^\w\-_]', '_', path)
    
    #