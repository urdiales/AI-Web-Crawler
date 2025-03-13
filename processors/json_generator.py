"""
JSON Generator

This module generates structured JSON from processed web content.
"""

import logging
import json
from typing import Dict, Any, List
import time
from datetime import datetime
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

def generate_json(pages: Dict[str, Dict[str, Any]], crawl_id: str, base_url: str) -> Dict[str, Any]:
    """
    Generate structured JSON from processed page data.
    
    Args:
        pages: Dictionary of processed pages
        crawl_id: Unique identifier for this crawl
        base_url: Base URL that was crawled
        
    Returns:
        Dictionary with structured JSON data
    """
    try:
        logger.info(f"Generating JSON for {len(pages)} pages")
        
        # Parse base domain
        parsed_url = urlparse(base_url)
        base_domain = parsed_url.netloc
        
        # Create the JSON structure
        json_data = {
            "crawl_id": crawl_id,
            "base_url": base_url,
            "domain": base_domain,
            "timestamp": int(time.time()),
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "page_count": len(pages),
            "pages": []
        }
        
        # Add pages
        for url, page in pages.items():
            # Create JSON for each page
            page_json = create_page_json(page)
            json_data["pages"].append(page_json)
        
        logger.info(f"JSON generation complete for {len(pages)} pages")
        return json_data
    
    except Exception as e:
        logger.error(f"Error generating JSON: {str(e)}", exc_info=True)
        # Return a minimal structure with error information
        return {
            "crawl_id": crawl_id,
            "base_url": base_url,
            "timestamp": int(time.time()),
            "error": str(e),
            "pages": []
        }

def create_page_json(page: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create JSON structure for a single page.
    
    Args:
        page: Dictionary with page information
        
    Returns:
        Dictionary with page JSON data
    """
    try:
        # Extract basic page information
        url = page.get('url', '')
        title = page.get('title', 'Untitled Page')
        metadata = page.get('metadata', {})
        depth = page.get('depth', 0)
        
        # Create the page JSON structure
        page_json = {
            "url": url,
            "title": title,
            "depth": depth,
            "metadata": metadata,
            "content": {
                "summary": page.get('summary', ''),
                "sections": []
            }
        }
        
        # Add sections
        if 'sections' in page and page['sections']:
            page_json["content"]["sections"] = page['sections']
        
        # Check if there's SharePoint list data
        if 'list_data' in page and page['list_data']:
            page_json["list_data"] = page['list_data']
            
        return page_json
    
    except Exception as e:
        logger.warning(f"Error creating page JSON: {str(e)}")
        return {
            "url": page.get('url', ''),
            "title": page.get('title', 'Error'),
            "error": str(e)
        }

def save_json_to_file(json_data: Dict[str, Any], filepath: str) -> bool:
    """
    Save JSON data to a file.
    
    Args:
        json_data: Dictionary with JSON data
        filepath: Path to save the file
        
    Returns:
        Boolean indicating success
    """
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        logger.info(f"JSON saved to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error saving JSON to file: {str(e)}", exc_info=True)
        return False