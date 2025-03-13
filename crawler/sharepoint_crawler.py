"""
SharePoint Crawler

This module provides specialized functionality for crawling SharePoint sites and lists.
It can handle SharePoint authentication and extract content from SharePoint-specific structures.
"""

import asyncio
import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from urllib.parse import urlparse
import json
import time

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from utils.error_handler import handle_network_error

logger = logging.getLogger(__name__)

def is_sharepoint_url(url: str) -> bool:
    """
    Check if a URL is a SharePoint site.
    
    Args:
        url: URL to check
        
    Returns:
        bool: True if it's a SharePoint URL
    """
    # Common SharePoint URL patterns
    sharepoint_patterns = [
        r'\.sharepoint\.com',
        r'/sites/',
        r'/teams/',
        r'/personal/',
        r'/_layouts/'
    ]
    
    for pattern in sharepoint_patterns:
        if re.search(pattern, url, re.IGNORECASE):
            return True
    
    return False

async def crawl_sharepoint(url: str, 
                          keywords: Optional[List[str]] = None,
                          include_images: bool = True) -> Dict[str, Any]:
    """
    Crawl a SharePoint site or list.
    
    Args:
        url: SharePoint URL to crawl
        keywords: Optional list of keywords to focus crawling
        include_images: Whether to include images
        
    Returns:
        Dict with crawl results
    """
    logger.info(f"Starting SharePoint crawl for {url}")
    start_time = time.time()
    
    try:
        # SharePoint requires authenticated sessions, use managed browser approach
        browser_config = BrowserConfig(
            headless=False,  # Initially visible for authentication
            user_data_dir="./data/browser_profile",
            use_managed_browser=True,
            verbose=True,
            timeout=120000  # Extended timeout for SharePoint
        )
        
        run_config = CrawlerRunConfig(
            wait_for_images=include_images,
            removed_overlay_elements=True,
            wait_for="css:.ms-List",  # Common SharePoint list element
            js_code=[
                # Script to expand SharePoint sections
                """(async () => {
                    // Expand any collapsed sections
                    const expandButtons = document.querySelectorAll('.ms-Button--icon:not([aria-expanded="true"])');
                    for (const button of expandButtons) {
                        button.click();
                        await new Promise(r => setTimeout(r, 1000));
                    }
                    
                    // Wait for content to load
                    await new Promise(r => setTimeout(r, 2000));
                    
                    // Click "show more" buttons if present
                    const showMoreButtons = Array.from(document.querySelectorAll('button'))
                        .filter(b => b.innerText && b.innerText.toLowerCase().includes('show more'));
                    for (const button of showMoreButtons) {
                        button.click();
                        await new Promise(r => setTimeout(r, 1000));
                    }
                })();"""
            ]
        )
        
        # First attempt with user authentication
        logger.info("Opening SharePoint site for authentication...")
        async with AsyncWebCrawler(config=browser_config) as crawler:
            # Allow time for user login (if needed)
            await asyncio.sleep(5)
            
            # Now run the actual crawl with the authenticated session
            browser_config.headless = True  # Switch to headless after authentication
            result = await crawler.arun(url=url, config=run_config)
            
            if not result.success:
                logger.error(f"SharePoint crawl failed: {result.error_message}")
                return {
                    "success": False,
                    "error": f"SharePoint crawl failed: {result.error_message}",
                    "url": url
                }
            
            # Process the SharePoint content
            pages = {}
            pages[url] = {
                "url": url,
                "title": _extract_sharepoint_title(result),
                "markdown": result.markdown.fit_markdown if hasattr(result.markdown, 'fit_markdown') else result.markdown,
                "html": result.cleaned_html,
                "depth": 0
            }
            
            # Extract SharePoint list data if present
            list_data = _extract_sharepoint_list(result.cleaned_html)
            if list_data:
                pages[url]["list_data"] = list_data
            
            # Extract images if needed
            images = []
            if include_images and hasattr(result, 'media'):
                images = result.media.get("images", [])
                for img in images:
                    img["page_url"] = url
            
            logger.info(f"SharePoint crawl completed successfully in {time.time() - start_time:.2f} seconds")
            
            return {
                "success": True,
                "url": url,
                "pages": pages,
                "images": images,
                "is_sharepoint": True,
                "duration_seconds": time.time() - start_time
            }
            
    except Exception as e:
        error_msg = f"Error during SharePoint crawl: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            "success": False,
            "url": url,
            "error": error_msg,
            "duration_seconds": time.time() - start_time
        }

def _extract_sharepoint_title(result) -> str:
    """Extract title from SharePoint page."""
    try:
        # Look for SharePoint-specific title elements
        if hasattr(result, 'cleaned_html') and result.cleaned_html:
            # Try SharePoint modern page title
            title_match = re.search(r'<div[^>]*class="[^"]*pageTitle[^"]*"[^>]*>(.*?)</div>', 
                                   result.cleaned_html, re.IGNORECASE | re.DOTALL)
            if title_match:
                return re.sub(r'<[^>]*>', '', title_match.group(1)).strip()
            
            # Try regular title tag
            title_match = re.search(r'<title>(.*?)</title>', result.cleaned_html, re.IGNORECASE)
            if title_match:
                return title_match.group(1).strip()
        
        # Try metadata
        if hasattr(result, 'metadata') and result.metadata and 'title' in result.metadata:
            return result.metadata['title']
        
        # Default to URL
        parsed_url = urlparse(result.url)
        path_parts = parsed_url.path.strip('/').split('/')
        if path_parts:
            return path_parts[-1].replace('-', ' ').replace('_', ' ').title()
        
        return "SharePoint Document"
    
    except Exception as e:
        logger.warning(f"Error extracting SharePoint title: {str(e)}")
        return "SharePoint Document"

def _extract_sharepoint_list(html: str) -> List[Dict[str, Any]]:
    """
    Extract data from a SharePoint list view.
    
    Args:
        html: HTML content of the SharePoint page
        
    Returns:
        List of dictionaries containing the list data, or empty list if not found
    """
    try:
        # Look for list data in the page
        # SharePoint modern lists use JSON data embedded in the page
        json_data_match = re.search(r'var g_listData\s*=\s*({.*?});', html, re.DOTALL)
        if json_data_match:
            try:
                list_data = json.loads(json_data_match.group(1))
                return list_data.get('Row', [])
            except json.JSONDecodeError:
                logger.warning("Could not parse SharePoint list JSON")
        
        # Try extracting from table structure if JSON approach failed
        items = []
        
        # Find table headers
        header_match = re.search(r'<tr[^>]*class="[^"]*ms-viewheadertr[^"]*"[^>]*>(.*?)</tr>', 
                                html, re.IGNORECASE | re.DOTALL)
        if not header_match:
            return []
            
        header_html = header_match.group(1)
        headers = re.findall(r'<th[^>]*>(.*?)</th>', header_html, re.IGNORECASE | re.DOTALL)
        
        if not headers:
            return []
            
        # Clean headers
        clean_headers = [re.sub(r'<[^>]*>', '', h).strip() for h in headers]
        
        # Find table rows
        rows = re.findall(r'<tr[^>]*class="[^"]*ms-itmhover[^"]*"[^>]*>(.*?)</tr>', 
                         html, re.IGNORECASE | re.DOTALL)
        
        for row_html in rows:
            cells = re.findall(r'<td[^>]*>(.*?)</td>', row_html, re.IGNORECASE | re.DOTALL)
            if len(cells) == len(clean_headers):
                item = {}
                for i, header in enumerate(clean_headers):
                    # Clean cell content
                    cell_content = re.sub(r'<[^>]*>', '', cells[i]).strip()
                    item[header] = cell_content
                items.append(item)
        
        return items
    
    except Exception as e:
        logger.warning(f"Error extracting SharePoint list: {str(e)}")
        return []