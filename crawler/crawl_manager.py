"""
Crawl Manager

This module coordinates the crawling process, delegating to specialized crawlers
and managing the crawl configuration.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from urllib.parse import urlparse
import re
from pathlib import Path
import json
import time

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.deep_crawling.filters import FilterChain, URLPatternFilter, DomainFilter

from .sharepoint_crawler import is_sharepoint_url, crawl_sharepoint
from .image_handler import process_images
from utils.error_handler import handle_network_error, handle_invalid_url
from utils.file_manager import save_to_file
from processors.content_processor import process_content
from processors.markdown_generator import generate_markdown
from processors.json_generator import generate_json
from database.vector_store import store_vectors
from database.embeddings import generate_embeddings

logger = logging.getLogger(__name__)

class CrawlManager:
    """
    Manages the web crawling process, coordinating between different specialized crawlers,
    handling configuration, and processing results.
    """
    
    def __init__(self, 
                 base_url: str,
                 keywords: Optional[List[str]] = None,
                 include_images: bool = True,
                 max_depth: int = 2,
                 max_pages: int = 50,
                 stay_within_domain: bool = True,
                 excluded_patterns: Optional[List[str]] = None):
        """
        Initialize the crawl manager.
        
        Args:
            base_url: Starting URL for crawling
            keywords: Optional list of keywords to focus crawling
            include_images: Whether to download and process images
            max_depth: Maximum depth for BFS crawling
            max_pages: Maximum number of pages to crawl
            stay_within_domain: Whether to stay within the initial domain
            excluded_patterns: URL patterns to exclude from crawling
        """
        self.base_url = base_url
        self.keywords = keywords if keywords else []
        self.include_images = include_images
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.stay_within_domain = stay_within_domain
        self.excluded_patterns = excluded_patterns if excluded_patterns else []
        
        # Add common excluded patterns
        self.excluded_patterns.extend([
            "*login*", "*signin*", "*logout*", 
            "*contact*", "*about*", "*career*",
            "*privacy*", "*terms*", "*cookie*"
        ])
        
        # Parse base domain for filtering
        parsed_url = urlparse(base_url)
        self.base_domain = parsed_url.netloc
        
        # Initialize result containers
        self.crawled_pages = {}
        self.crawled_images = []
        self.errors = []
        
        logger.info(f"CrawlManager initialized for {base_url} (max depth: {max_depth}, max pages: {max_pages})")
        
    async def start_crawl(self) -> Dict[str, Any]:
        """
        Start the crawling process.
        
        Returns:
            Dict containing crawl results and statistics
        """
        start_time = time.time()
        
        try:
            # Validate URL
            if not self._validate_url(self.base_url):
                error_msg = f"Invalid URL: {self.base_url}"
                logger.error(error_msg)
                return {"success": False, "error": error_msg}
            
            # Check if it's a SharePoint site
            if is_sharepoint_url(self.base_url):
                logger.info(f"Detected SharePoint URL: {self.base_url}")
                return await crawl_sharepoint(self.base_url, self.keywords, self.include_images)
            
            # Standard web crawling
            crawl_results = await self._perform_web_crawl()
            
            # Process and store results
            if crawl_results["success"]:
                # Generate and store JSON and Markdown
                json_path = await self._process_and_store_results(crawl_results["pages"])
                
                # Process images if enabled
                image_results = {}
                if self.include_images and crawl_results["images"]:
                    image_results = await process_images(crawl_results["images"])
                
                # Create embeddings and store vectors
                embedding_results = await self._create_embeddings(crawl_results["pages"])
                
                # Prepare final results
                duration = time.time() - start_time
                return {
                    "success": True,
                    "url": self.base_url,
                    "pages_count": len(crawl_results["pages"]),
                    "images_count": len(image_results.get("processed_images", [])),
                    "duration_seconds": duration,
                    "json_path": json_path,
                    "errors": self.errors + crawl_results.get("errors", []),
                    "pages": crawl_results["pages"],
                    "images": image_results.get("processed_images", [])
                }
            else:
                return crawl_results  # Return the error results
                
        except Exception as e:
            logger.error(f"Error during crawl: {str(e)}", exc_info=True)
            return {
                "success": False,
                "url": self.base_url,
                "error": str(e),
                "duration_seconds": time.time() - start_time
            }
    
    async def _perform_web_crawl(self) -> Dict[str, Any]:
        """
        Execute the web crawling using Crawl4AI.
        
        Returns:
            Dict containing crawl results
        """
        try:
            logger.info(f"Starting web crawl of {self.base_url}")
            
            # Create URL patterns from keywords if provided
            keyword_patterns = []
            if self.keywords and len(self.keywords) > 0:
                keyword_patterns = [f"*{keyword}*" for keyword in self.keywords]
            
            # Set up filter chain
            filters = []
            
            # Add domain filter if staying within domain
            if self.stay_within_domain:
                domain_filter = DomainFilter(
                    allowed_domains=[self.base_domain],
                    blocked_domains=[]
                )
                filters.append(domain_filter)
            
            # Add URL pattern filter if we have patterns
            if keyword_patterns or self.excluded_patterns:
                url_filter = URLPatternFilter(
                    patterns=keyword_patterns,
                    excluded_patterns=self.excluded_patterns
                )
                filters.append(url_filter)
            
            filter_chain = FilterChain(filters) if filters else None
            
            # Set up deep crawl strategy
            deep_crawl_strategy = BFSDeepCrawlStrategy(
                max_depth=self.max_depth,
                include_external=not self.stay_within_domain,
                max_pages=self.max_pages,
                filter_chain=filter_chain
            )
            
            # Configure content filter for meaningful content
            content_filter = PruningContentFilter(
                threshold=0.48,
                threshold_type="fixed",
                min_word_threshold=20  # Ignore very small blocks
            )
            
            # Configure crawler
            browser_config = BrowserConfig(
                headless=True,
                verbose=True
            )
            
            run_config = CrawlerRunConfig(
                deep_crawl_strategy=deep_crawl_strategy,
                cache_mode=CacheMode.ENABLED,
                markdown_generator=DefaultMarkdownGenerator(
                    content_filter=content_filter
                ),
                excluded_tags=['nav', 'footer', 'header', 'aside', 'menu'],
                exclude_external_links=self.stay_within_domain,
                exclude_social_media_links=True,
                wait_for_images=self.include_images,
                scan_full_page=True,
                scroll_delay=0.5
            )
            
            # Execute crawl
            async with AsyncWebCrawler(config=browser_config) as crawler:
                results = await crawler.arun(
                    url=self.base_url,
                    config=run_config
                )
                
                # Process results
                pages = {}
                images = []
                errors = []
                
                # If it's a deep crawl, results will be a list
                if isinstance(results, list):
                    for result in results:
                        if result.success:
                            pages[result.url] = {
                                "url": result.url,
                                "title": self._extract_title(result),
                                "markdown": result.markdown.fit_markdown,
                                "html": result.cleaned_html,
                                "depth": result.metadata.get("depth", 0)
                            }
                            # Collect images
                            if self.include_images and hasattr(result, 'media'):
                                page_images = result.media.get("images", [])
                                for img in page_images:
                                    img["page_url"] = result.url
                                    images.append(img)
                        else:
                            errors.append({
                                "url": result.url,
                                "error": result.error_message
                            })
                else:
                    # Single page result
                    if results.success:
                        pages[results.url] = {
                            "url": results.url,
                            "title": self._extract_title(results),
                            "markdown": results.markdown.fit_markdown,
                            "html": results.cleaned_html,
                            "depth": 0
                        }
                        # Collect images
                        if self.include_images and hasattr(results, 'media'):
                            page_images = results.media.get("images", [])
                            for img in page_images:
                                img["page_url"] = results.url
                                images.append(img)
                    else:
                        errors.append({
                            "url": results.url, 
                            "error": results.error_message
                        })
                
                logger.info(f"Web crawl completed. Found {len(pages)} pages and {len(images)} images.")
                return {
                    "success": True,
                    "pages": pages,
                    "images": images,
                    "errors": errors
                }
                
        except Exception as e:
            error_msg = f"Error during web crawl: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                "success": False,
                "error": error_msg
            }
    
    async def _process_and_store_results(self, pages: Dict[str, Any]) -> str:
        """
        Process crawled content and store as JSON and Markdown.
        
        Args:
            pages: Dictionary of crawled pages
            
        Returns:
            Path to the generated JSON file
        """
        try:
            # Generate a timestamp-based ID for this crawl
            timestamp = int(time.time())
            domain = self.base_domain.replace(".", "_")
            crawl_id = f"{domain}_{timestamp}"
            
            # Process content using the processor
            processed_pages = {}
            for url, page in pages.items():
                processed_page = process_content(page)
                processed_pages[url] = processed_page
            
            # Generate JSON
            json_data = generate_json(processed_pages, crawl_id, self.base_url)
            
            # Save JSON file
            json_filename = f"{crawl_id}.json"
            json_path = Path("data/json") / json_filename
            
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved JSON data to {json_path}")
            
            # Generate and save Markdown files
            markdown_dir = Path("data/markdown") / crawl_id
            markdown_dir.mkdir(exist_ok=True, parents=True)
            
            for url, page in processed_pages.items():
                markdown_content = generate_markdown(page)
                filename = self._url_to_filename(url)
                markdown_path = markdown_dir / f"{filename}.md"
                
                with open(markdown_path, "w", encoding="utf-8") as f:
                    f.write(markdown_content)
            
            logger.info(f"Saved Markdown files to {markdown_dir}")
            
            return str(json_path)
            
        except Exception as e:
            error_msg = f"Error processing and storing results: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.errors.append({"type": "processing_error", "message": error_msg})
            return ""
    
    async def _create_embeddings(self, pages: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create embeddings from processed content and store in vector database.
        
        Args:
            pages: Dictionary of crawled pages
            
        Returns:
            Dict with embedding results
        """
        try:
            # Generate embeddings for each page
            page_contents = [page["markdown"] for page in pages.values()]
            page_urls = list(pages.keys())
            
            embeddings = await generate_embeddings(page_contents)
            
            # Store vectors in the database
            vector_results = await store_vectors(embeddings, page_urls, page_contents)
            
            logger.info(f"Created and stored embeddings for {len(embeddings)} pages")
            return {
                "success": True,
                "vector_count": len(embeddings)
            }
            
        except Exception as e:
            error_msg = f"Error creating embeddings: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.errors.append({"type": "embedding_error", "message": error_msg})
            return {
                "success": False,
                "error": error_msg
            }
    
    def _validate_url(self, url: str) -> bool:
        """Validate URL format."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception as e:
            logger.error(f"URL validation error: {str(e)}")
            return False
    
    def _extract_title(self, result) -> str:
        """Extract page title from crawl result."""
        try:
            # Try to get from metadata
            if hasattr(result, 'metadata') and result.metadata and 'title' in result.metadata:
                return result.metadata['title']
            
            # Otherwise extract from HTML
            if hasattr(result, 'cleaned_html') and result.cleaned_html:
                title_match = re.search("<title>(.*?)</title>", result.cleaned_html, re.IGNORECASE)
                if title_match:
                    return title_match.group(1)
            
            # Fallback to URL
            parsed_url = urlparse(result.url)
            path = parsed_url.path
            if path and path != "/":
                # Get last part of the path
                path_parts = path.rstrip("/").split("/")
                return path_parts[-1].replace("-", " ").replace("_", " ").title()
            else:
                return parsed_url.netloc
                
        except Exception as e:
            logger.warning(f"Error extracting title: {str(e)}")
            return "Untitled Page"
    
    def _url_to_filename(self, url: str) -> str:
        """Convert URL to a valid filename."""
        parsed = urlparse(url)
        path = parsed.path.rstrip("/")
        if not path:
            path = "index"
        else:
            # Remove leading slash and replace special chars
            path = path.lstrip("/")
            
        # Replace special characters
        filename = re.sub(r'[^\w\-_]', '_', path)
        
        # Truncate if too long
        if len(filename) > 100:
            filename = filename[:100]
            
        return filename