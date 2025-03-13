"""
Crawl UI

This module handles the web crawler UI component.
"""

import streamlit as st
import asyncio
import logging
import time
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import base64

from crawler.crawl_manager import CrawlManager
from utils.file_manager import json_to_markdown, create_download_link
from utils.error_handler import handle_invalid_url, handle_error
from ui.dashboard import display_footer

logger = logging.getLogger(__name__)

def render_crawl_interface():
    """Render the web crawler interface."""
    try:
        st.header("Web Crawler")
        st.write("Enter a URL to crawl and extract knowledge.")
        
        # URL input
        url = st.text_input(
            "URL to crawl",
            placeholder="https://example.com",
            key="crawl_url"
        )
        
        # Advanced options (collapsible)
        with st.expander("Advanced Options"):
            col1, col2 = st.columns(2)
            
            with col1:
                keywords = st.text_area(
                    "Focus Keywords (optional)",
                    placeholder="Enter keywords (one per line)",
                    help="The crawler will prioritize pages containing these keywords",
                    key="keywords"
                )
                
                include_images = st.checkbox(
                    "Download Images",
                    value=True,
                    help="Download and store images from the website",
                    key="include_images"
                )
            
            with col2:
                max_depth = st.slider(
                    "Crawl Depth",
                    min_value=1,
                    max_value=5,
                    value=2,
                    help="Maximum depth of pages to crawl",
                    key="max_depth"
                )
                
                max_pages = st.slider(
                    "Maximum Pages",
                    min_value=5,
                    max_value=100,
                    value=30,
                    help="Maximum number of pages to crawl",
                    key="max_pages"
                )
                
                stay_within_domain = st.checkbox(
                    "Stay Within Domain",
                    value=True,
                    help="Only crawl pages within the same domain",
                    key="stay_within_domain"
                )
        
        # Submit button
        if st.button("Start Crawl", type="primary", key="start_crawl"):
            if not url or not url.startswith(("http://", "https://")):
                st.error("Please enter a valid URL starting with http:// or https://")
                return
            
            # Parse keywords
            keywords_list = []
            if keywords:
                keywords_list = [k.strip() for k in keywords.split("\n") if k.strip()]
            
            # Start crawl
            asyncio.run(start_crawl(
                url, 
                keywords_list, 
                include_images, 
                max_depth, 
                max_pages, 
                stay_within_domain
            ))
        
        # Show recent crawls
        display_recent_crawls()
        
        # Footer
        display_footer()
    
    except Exception as e:
        logger.error(f"Error rendering crawl interface: {str(e)}", exc_info=True)
        st.error(f"An error occurred: {str(e)}")

async def start_crawl(url: str, 
                     keywords: List[str], 
                     include_images: bool,
                     max_depth: int,
                     max_pages: int,
                     stay_within_domain: bool):
    """
    Start the crawl process.
    
    Args:
        url: URL to crawl
        keywords: List of focus keywords
        include_images: Whether to download images
        max_depth: Maximum crawl depth
        max_pages: Maximum number of pages
        stay_within_domain: Whether to stay within the same domain
    """
    try:
        # Show progress
        progress_container = st.empty()
        progress_container.info("Initializing crawler...")
        
        # Create crawl manager
        crawl_manager = CrawlManager(
            base_url=url,
            keywords=keywords,
            include_images=include_images,
            max_depth=max_depth,
            max_pages=max_pages,
            stay_within_domain=stay_within_domain
        )
        
        # Start crawl
        progress_container.info(f"Crawling {url}...")
        results = await crawl_manager.start_crawl()
        
        if not results["success"]:
            progress_container.error(f"Crawl failed: {results.get('error', 'Unknown error')}")
            return
        
        # Show results
        progress_container.success(f"Crawl completed! Processed {results['pages_count']} pages.")
        display_crawl_results(results)
    
    except Exception as e:
        logger.error(f"Error starting crawl: {str(e)}", exc_info=True)
        st.error(f"An error occurred during crawl: {str(e)}")

def display_crawl_results(results: Dict[str, Any]):
    """
    Display the results of a crawl.
    
    Args:
        results: Crawl results dictionary
    """
    try:
        st.subheader("Crawl Results")
        
        # Summary stats
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Pages Crawled",
                value=results["pages_count"]
            )
        
        with col2:
            st.metric(
                label="Images",
                value=results.get("images_count", 0)
            )
        
        with col3:
            st.metric(
                label="Time",
                value=f"{results.get('duration_seconds', 0):.1f}s"
            )
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["Overview", "JSON", "Markdown", "Images"])
        
        with tab1:
            display_crawl_overview(results)
        
        with tab2:
            display_json_view(results)
        
        with tab3:
            display_markdown_view(results)
        
        with tab4:
            display_images_view(results)
    
    except Exception as e:
        logger.error(f"Error displaying crawl results: {str(e)}", exc_info=True)
        st.error(f"Error displaying results: {str(e)}")

def display_crawl_overview(results: Dict[str, Any]):
    """
    Display an overview of the crawl results.
    
    Args:
        results: Crawl results dictionary
    """
    try:
        # Basic info
        st.write(f"**URL:** {results['url']}")
        st.write(f"**Time:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Pages table
        if "pages" in results and results["pages"]:
            st.subheader("Crawled Pages")
            
            # Convert dict to list
            pages_list = []
            for url, page in results["pages"].items():
                pages_list.append({
                    "URL": url,
                    "Title": page.get("title", "Untitled"),
                    "Depth": page.get("depth", 0)
                })
            
            # Sort by depth
            pages_list.sort(key=lambda x: (x["Depth"], x["Title"]))
            
            # Display as table
            st.dataframe(
                pages_list,
                column_config={
                    "URL": st.column_config.LinkColumn("URL"),
                    "Title": st.column_config.TextColumn("Title"),
                    "Depth": st.column_config.NumberColumn("Depth")
                },
                hide_index=True
            )
        
        # Errors
        if "errors" in results and results["errors"]:
            st.subheader("Errors")
            
            for error in results["errors"]:
                st.error(f"{error.get('url', 'Unknown URL')}: {error.get('error', 'Unknown error')}")
    
    except Exception as e:
        logger.error(f"Error displaying crawl overview: {str(e)}", exc_info=True)
        st.error(f"Error displaying overview: {str(e)}")

def display_json_view(results: Dict[str, Any]):
    """
    Display and allow download of JSON data.
    
    Args:
        results: Crawl results dictionary
    """
    try:
        if "json_path" in results and results["json_path"]:
            json_path = results["json_path"]
            
            # Read JSON file
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # Display JSON preview
            st.json(json_data, expanded=False)
            
            # Download button
            json_str = json.dumps(json_data, indent=2)
            filename = os.path.basename(json_path)
            
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name=filename,
                mime="application/json",
                key="download_json"
            )
    
    except Exception as e:
        logger.error(f"Error displaying JSON view: {str(e)}", exc_info=True)
        st.error(f"Error displaying JSON data: {str(e)}")

def display_markdown_view(results: Dict[str, Any]):
    """
    Display and generate Markdown files.
    
    Args:
        results: Crawl results dictionary
    """
    try:
        if "json_path" in results and results["json_path"]:
            json_path = results["json_path"]
            
            # Generate markdown button
            if st.button("Generate Markdown Files", key="generate_markdown"):
                # Convert JSON to Markdown
                with st.spinner("Generating Markdown files..."):
                    markdown_results = asyncio.run(json_to_markdown(json_path))
                
                if markdown_results["success"]:
                    st.success(f"Generated {markdown_results['file_count']} Markdown files")
                    st.write(f"Files saved to: `{markdown_results['output_dir']}`")
                    
                    # Show sample
                    if markdown_results.get("files"):
                        sample_file = markdown_results["files"][0]
                        
                        with open(sample_file["path"], 'r', encoding='utf-8') as f:
                            sample_content = f.read()
                        
                        st.subheader(f"Sample: {sample_file['title']}")
                        st.code(sample_content, language="markdown")
                else:
                    st.error(f"Error generating Markdown: {markdown_results.get('error', 'Unknown error')}")
    
    except Exception as e:
        logger.error(f"Error displaying Markdown view: {str(e)}", exc_info=True)
        st.error(f"Error handling Markdown generation: {str(e)}")

def display_images_view(results: Dict[str, Any]):
    """
    Display crawled images.
    
    Args:
        results: Crawl results dictionary
    """
    try:
        if "images" in results and results["images"]:
            images = results["images"]
            
            if not images:
                st.info("No images were crawled.")
                return
            
            st.write(f"Found {len(images)} images")
            
            # Group images by source page
            images_by_page = {}
            for img in images:
                page_url = img.get("page_url", "Unknown")
                if page_url not in images_by_page:
                    images_by_page[page_url] = []
                images_by_page[page_url].append(img)
            
            # Display images grouped by page
            for page_url, page_images in images_by_page.items():
                with st.expander(f"{page_url} ({len(page_images)} images)"):
                    image_grid(page_images)
    
    except Exception as e:
        logger.error(f"Error displaying images view: {str(e)}", exc_info=True)
        st.error(f"Error displaying images: {str(e)}")

def image_grid(images: List[Dict[str, Any]], cols: int = 3):
    """
    Display images in a grid.
    
    Args:
        images: List of image dictionaries
        cols: Number of columns in the grid
    """
    # Calculate rows needed
    rows = (len(images) + cols - 1) // cols
    
    for row in range(rows):
        columns = st.columns(cols)
        
        for col in range(cols):
            idx = row * cols + col
            if idx < len(images):
                img = images[idx]
                
                with columns[col]:
                    # Display image if local path exists
                    if "local_path" in img and os.path.exists(img["local_path"]):
                        st.image(
                            img["local_path"],
                            caption=img.get("alt", ""),
                            width=200
                        )
                    # Otherwise try to display from src URL
                    elif "src" in img:
                        st.image(
                            img["src"],
                            caption=img.get("alt", ""),
                            width=200
                        )

def display_recent_crawls():
    """Display a list of recent crawls."""
    try:
        # Check if JSON directory exists
        json_dir = Path("data/json")
        if not json_dir.exists() or not json_dir.is_dir():
            return
        
        # Get all JSON files
        json_files = list(json_dir.glob("*.json"))
        if not json_files:
            return
        
        # Sort by modification time (most recent first)
        json_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        # Display recent crawls
        st.subheader("Recent Crawls")
        
        for file_path in json_files[:5]:  # Show 5 most recent
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extract basic information
                base_url = data.get("base_url", "Unknown")
                domain = data.get("domain", "Unknown")
                date = data.get("date", "Unknown")
                page_count = data.get("page_count", 0)
                
                with st.expander(f"{domain} - {date} ({page_count} pages)"):
                    st.write(f"**URL:** {base_url}")
                    st.write(f"**File:** {file_path}")
                    
                    # Add button to load this crawl
                    if st.button("Load Crawl", key=f"load_{file_path.stem}"):
                        # Read JSON file
                        with open(file_path, 'r', encoding='utf-8') as f:
                            json_data = json.load(f)
                        
                        # Convert pages format from JSON to results format
                        pages = {}
                        for page in json_data.get("pages", []):
                            url = page.get("url", "")
                            if url:
                                pages[url] = page
                        
                        # Create minimal results dict
                        results = {
                            "success": True,
                            "url": base_url,
                            "pages_count": page_count,
                            "pages": pages,
                            "json_path": str(file_path),
                            "images_count": 0,
                            "images": []
                        }
                        
                        # Display results
                        display_crawl_results(results)
            
            except Exception as e:
                logger.warning(f"Error loading crawl {file_path}: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error displaying recent crawls: {str(e)}", exc_info=True)