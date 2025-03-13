"""
Web Crawler Module

Handles crawling websites using Crawl4AI with support for deep crawling,
domain filtering, and keyword relevance scoring.
"""

import asyncio
import os
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any, Union
from datetime import datetime
import json

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.deep_crawling.filters import FilterChain, URLPatternFilter, DomainFilter
from crawl4ai.deep_crawling.scorers import KeywordRelevanceScorer
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from crawl4ai.content_filter_strategy import PruningContentFilter

from config import SETTINGS
from utils import setup_logger

# Set up logger
logger = setup_logger("crawler")

class WebCrawler:
    """
    Web crawler implementation using Crawl4AI.
    """
    
    def __init__(
        self,
        max_depth: int = 2,
        include_images: bool = True,
        stay_within_domain: bool = True,
        max_pages: int = 50,
        keywords: Optional[List[str]] = None,
        word_count_threshold: int = 10,
        excluded_tags: List[str] = None
    ):
        """
        Initialize the WebCrawler.
        
        Args:
            max_depth: Depth of crawl (0 for initial page only)
            include_images: Whether to include images in results
            stay_within_domain: Whether to stay within the same domain
            max_pages: Maximum number of pages to crawl
            keywords: Optional list of keywords to focus on
            word_count_threshold: Minimum words for content blocks
            excluded_tags: HTML tags to exclude (defaults to nav, footer, etc.)
        """
        self.max_depth = max_depth
        self.include_images = include_images
        self.stay_within_domain = stay_within_domain
        self.max_pages = max_pages
        self.keywords = keywords
        self.word_count_threshold = word_count_threshold
        
        if excluded_tags is None:
            self.excluded_tags = ['nav', 'footer', 'header', 'aside', 'form', 'script', 'style']
        else:
            self.excluded_tags = excluded_tags
        
        # Will be initialized before crawling
        self.browser_config = None
        self.crawler_run_config = None
    
    def _prepare_config(self, url: str) -> None:
        """
        Prepare browser and crawler configurations.
        
        Args:
            url: The target URL to prepare configs for
        """
        # Set up browser config
        self.browser_config = BrowserConfig(
            headless=True,
            verbose=SETTINGS.DEBUG,
            user_agent=SETTINGS.USER_AGENT,
            browser_type="chromium",
            page_timeout=60000  # 60 seconds
        )
        
        # Handle SharePoint/Active Directory if needed
        if "sharepoint" in url.lower():
            # Use persistent context for SharePoint
            user_data_dir = os.path.join(Path.home(), ".crawl4ai", "sharepoint_profile")
            os.makedirs(user_data_dir, exist_ok=True)
            
            self.browser_config.user_data_dir = user_data_dir
            self.browser_config.use_persistent_context = True
        
        # Set up BFS crawl strategy
        url_filters = []
        
        # Domain filter if staying within domain
        if self.stay_within_domain:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc
            url_filters.append(DomainFilter(allowed_domains=[domain]))
        
        # URL patterns if using keywords
        if self.keywords and len(self.keywords) > 0:
            patterns = [f"*{keyword.lower()}*" for keyword in self.keywords]
            url_filters.append(URLPatternFilter(patterns=patterns))
        
        # Create filter chain if needed
        filter_chain = FilterChain(url_filters) if url_filters else None
        
        # Create scorer if using keywords
        url_scorer = None
        if self.keywords and len(self.keywords) > 0:
            url_scorer = KeywordRelevanceScorer(keywords=self.keywords, weight=0.7)
        
        # Create deep crawl strategy
        deep_crawl_strategy = BFSDeepCrawlStrategy(
            max_depth=self.max_depth,
            include_external=not self.stay_within_domain,
            max_pages=self.max_pages,
            filter_chain=filter_chain,
            url_scorer=url_scorer
        )
        
        # Create the crawler config
        self.crawler_run_config = CrawlerRunConfig(
            deep_crawl_strategy=deep_crawl_strategy,
            scraping_strategy=LXMLWebScrapingStrategy(),
            wait_for_images=self.include_images,
            excluded_tags=self.excluded_tags,
            word_count_threshold=self.word_count_threshold,
            exclude_social_media_links=True,
            remove_overlay_elements=True,
            wait_for=".main-content, main, #main, article",  # Common content containers
            stream=True,  # Enable streaming for progress updates
            verbose=SETTINGS.DEBUG
        )
    
    async def _handle_sharepoint_auth(self, crawler, url: str) -> bool:
        """
        Handle SharePoint authentication if needed.
        
        Args:
            crawler: The AsyncWebCrawler instance
            url: The SharePoint URL
            
        Returns:
            bool: Whether authentication was handled successfully
        """
        # This is a placeholder for SharePoint authentication
        # In a real implementation, you would use Office365-REST-Python-Client
        # or msal library to handle the authentication flow
        
        if "sharepoint" not in url.lower():
            return True
        
        try:
            # For SharePoint, we might need to wait for login forms and inject credentials
            # This is simplified and would need to be expanded for real use cases
            if SETTINGS.SHAREPOINT_USERNAME and SETTINGS.SHAREPOINT_PASSWORD:
                logger.info("Handling SharePoint authentication...")
                
                # Example JS to handle login form
                js_code = f"""
                (async () => {{
                    // Look for username field
                    const usernameField = document.querySelector('input[type="email"], input[name="loginfmt"]');
                    if (usernameField) {{
                        usernameField.value = "{SETTINGS.SHAREPOINT_USERNAME}";
                        
                        // Find and click the next button
                        const nextButton = Array.from(document.querySelectorAll('button')).find(
                            button => button.textContent.includes('Next') || button.textContent.includes('Sign in')
                        );
                        if (nextButton) {{
                            nextButton.click();
                            await new Promise(r => setTimeout(r, 2000));
                        }}
                    }}
                    
                    // Look for password field
                    const passwordField = document.querySelector('input[type="password"]');
                    if (passwordField) {{
                        passwordField.value = "{SETTINGS.SHAREPOINT_PASSWORD}";
                        
                        // Find and click the sign-in button
                        const signInButton = Array.from(document.querySelectorAll('button')).find(
                            button => button.textContent.includes('Sign in') || button.textContent.includes('Submit')
                        );
                        if (signInButton) {{
                            signInButton.click();
                        }}
                    }}
                }})();
                """
                
                # Create a simple config for auth handling
                auth_config = CrawlerRunConfig(
                    js_code=js_code,
                    wait_for=10,  # Wait 10 seconds after executing the code
                    verbose=True
                )
                
                # Try to authenticate
                result = await crawler.arun(url=url, config=auth_config)
                return result.success
                
            else:
                logger.warning("SharePoint credentials not provided in settings")
                return False
        
        except Exception as e:
            logger.error(f"Error during SharePoint authentication: {e}")
            return False
    
    def _process_crawl_results(self, results: List[Any]) -> Dict[str, Any]:
        """
        Process raw crawl results into a structured format.
        
        Args:
            results: List of CrawlResult objects from Crawl4AI
            
        Returns:
            Dict containing structured crawl data
        """
        processed_data = {
            "metadata": {
                "crawl_date": datetime.now().isoformat(),
                "url_count": len(results),
                "keywords": self.keywords,
                "max_depth": self.max_depth
            },
            "pages": []
        }
        
        for result in results:
            if not result.success:
                logger.warning(f"Failed to crawl {result.url}: {result.error_message}")
                continue
            
            page_data = {
                "url": result.url,
                "title": result.metadata.get("title", "Untitled"),
                "content": result.markdown.fit_markdown if hasattr(result.markdown, "fit_markdown") else result.markdown,
                "depth": result.metadata.get("depth", 0),
                "score": result.metadata.get("score", 0),
                "crawl_time": result.metadata.get("crawl_time", 0)
            }
            
            # Add images if included
            if self.include_images and hasattr(result, "media") and result.media:
                images = result.media.get("images", [])
                if images:
                    page_data["images"] = []
                    for img in images:
                        img_data = {
                            "src": img.get("src", ""),
                            "alt": img.get("alt", ""),
                            "width": img.get("width", 0),
                            "height": img.get("height", 0)
                        }
                        page_data["images"].append(img_data)
            
            processed_data["pages"].append(page_data)
        
        return processed_data
    
    async def _crawl_async(self, url: str, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Execute the actual crawling operation asynchronously.
        
        Args:
            url: The URL to crawl
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dict containing structured crawl data
        """
        # Prepare configurations
        self._prepare_config(url)
        
        try:
            # Start the crawler
            async with AsyncWebCrawler(config=self.browser_config) as crawler:
                # Handle SharePoint authentication if needed
                if "sharepoint" in url.lower():
                    auth_success = await self._handle_sharepoint_auth(crawler, url)
                    if not auth_success:
                        logger.error("SharePoint authentication failed")
                        return {"error": "SharePoint authentication failed", "pages": []}
                
                # Start the crawl
                results = []
                total_pages = 0
                current_page = 0
                
                # Get initial async iterator
                result_iterator = await crawler.arun(url=url, config=self.crawler_run_config)
                
                # Process each result as it comes in
                async for result in result_iterator:
                    results.append(result)
                    current_page += 1
                    
                    # Update progress if callback provided
                    if progress_callback:
                        # If we know max_pages, use it for progress
                        if total_pages == 0 and hasattr(result.metadata, "total_pages"):
                            total_pages = result.metadata.get("total_pages", self.max_pages)
                        
                        # Otherwise use max_pages setting
                        if total_pages == 0:
                            total_pages = self.max_pages
                        
                        progress_callback(
                            current=current_page,
                            total=total_pages,
                            message=f"Crawling page: {result.url}"
                        )
                
                # Process the results
                return self._process_crawl_results(results)
        
        except Exception as e:
            logger.error(f"Error during crawling: {e}", exc_info=True)
            return {"error": str(e), "pages": []}
    
    def crawl(self, url: str, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Crawl a website and return structured data.
        
        Args:
            url: The URL to crawl
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dict containing structured crawl data
        """
        return asyncio.run(self._crawl_async(url, progress_callback))