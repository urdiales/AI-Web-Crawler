"""
Content Processor

This module processes raw HTML content into clean, structured content
suitable for RAG systems.
"""

import re
import logging
from bs4 import BeautifulSoup
from typing import Dict, Any, List
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

def process_content(page_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process the raw HTML content into clean, structured content.
    
    Args:
        page_data: Dictionary with page information
        
    Returns:
        Updated page data with processed content
    """
    try:
        # Extract URL and title
        url = page_data.get('url', '')
        title = page_data.get('title', 'Untitled Page')
        html = page_data.get('html', '')
        markdown = page_data.get('markdown', '')
        
        # Skip if no HTML or markdown content
        if not html and not markdown:
            logger.warning(f"No content to process for {url}")
            return page_data
        
        # Create the processed page data
        processed_data = {
            **page_data,
            "processed": True,
            "sections": []
        }
        
        # Extract metadata
        metadata = extract_metadata(html)
        processed_data["metadata"] = metadata
        
        # Split content into sections
        if html:
            sections = extract_sections(html, title)
            processed_data["sections"] = sections
        
        # Generate summary (first 200 words or so)
        if markdown:
            # Strip markdown formatting
            text_only = re.sub(r'#+ ', '', markdown)  # Remove headers
            text_only = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text_only)  # Replace links with text
            text_only = re.sub(r'```.*?```', '', text_only, flags=re.DOTALL)  # Remove code blocks
            text_only = re.sub(r'`.*?`', '', text_only)  # Remove inline code
            
            # Get first 200 words or so
            words = text_only.split()
            summary = ' '.join(words[:200])
            if len(words) > 200:
                summary += '...'
                
            processed_data["summary"] = summary
        
        # Extract domain
        parsed_url = urlparse(url)
        processed_data["domain"] = parsed_url.netloc
        
        logger.info(f"Processed content for {url}")
        return processed_data
    
    except Exception as e:
        logger.error(f"Error processing content: {str(e)}", exc_info=True)
        # Return original data plus error
        return {
            **page_data,
            "processed": False,
            "processing_error": str(e)
        }

def extract_metadata(html: str) -> Dict[str, Any]:
    """
    Extract metadata from HTML content.
    
    Args:
        html: HTML content
        
    Returns:
        Dictionary with metadata
    """
    try:
        soup = BeautifulSoup(html, 'html.parser')
        metadata = {}
        
        # Extract title
        title_tag = soup.find('title')
        if title_tag:
            metadata['title'] = title_tag.text.strip()
        
        # Extract meta tags
        for meta in soup.find_all('meta'):
            name = meta.get('name', meta.get('property', ''))
            content = meta.get('content', '')
            
            if name and content:
                # Clean up name
                name = name.lower().replace(':', '_')
                metadata[name] = content
        
        # Common meta tags to look for
        key_meta_tags = {
            'description': 'description',
            'keywords': 'keywords',
            'author': 'author',
            'og_title': 'og:title',
            'og_description': 'og:description',
            'twitter_title': 'twitter:title',
            'twitter_description': 'twitter:description'
        }
        
        # Ensure we have at least empty values for common tags
        for key, tag in key_meta_tags.items():
            if tag.replace(':', '_') not in metadata:
                metadata[tag.replace(':', '_')] = ''
        
        return metadata
    
    except Exception as e:
        logger.warning(f"Error extracting metadata: {str(e)}")
        return {}

def extract_sections(html: str, title: str) -> List[Dict[str, Any]]:
    """
    Extract logical sections from HTML content.
    
    Args:
        html: HTML content
        title: Page title
        
    Returns:
        List of section dictionaries
    """
    try:
        soup = BeautifulSoup(html, 'html.parser')
        sections = []
        
        # First section is always the title/intro
        intro_section = {
            "id": "section-0",
            "title": title,
            "level": 1,
            "content": ""
        }
        
        # Find all headings
        headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        
        # If no headings, treat the whole document as one section
        if not headings:
            # Get the main content
            main_content = soup.find('body')
            if main_content:
                intro_section["content"] = main_content.get_text().strip()
            sections.append(intro_section)
            return sections
        
        # Process headings into sections
        current_section = intro_section
        sections.append(current_section)
        
        for i, heading in enumerate(headings):
            # Determine heading level
            level = int(heading.name[1])
            
            # Create new section
            section = {
                "id": f"section-{i+1}",
                "title": heading.get_text().strip(),
                "level": level,
                "content": ""
            }
            
            # Collect content for this section (everything until next heading)
            content_parts = []
            for sibling in heading.next_siblings:
                if sibling.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    break
                if sibling.string:
                    content_parts.append(sibling.string)
                elif hasattr(sibling, 'get_text'):
                    content_parts.append(sibling.get_text())
            
            section["content"] = ' '.join(content_parts).strip()
            sections.append(section)
        
        return sections
    
    except Exception as e:
        logger.warning(f"Error extracting sections: {str(e)}")
        return [{
            "id": "section-0",
            "title": title,
            "level": 1,
            "content": "Error extracting sections"
        }]

def clean_html(html: str) -> str:
    """
    Clean HTML content, removing scripts, styles, and unnecessary elements.
    
    Args:
        html: HTML content
        
    Returns:
        Cleaned HTML
    """
    try:
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove scripts, styles, and comments
        for element in soup(["script", "style"]):
            element.decompose()
        
        # Remove comments
        for comment in soup.find_all(text=lambda text: isinstance(text, str) and text.strip().startswith('<!--')):
            comment.extract()
        
        # Clean up the HTML
        clean_html = str(soup)
        
        return clean_html
    
    except Exception as e:
        logger.warning(f"Error cleaning HTML: {str(e)}")
        return html