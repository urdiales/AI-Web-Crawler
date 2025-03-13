"""
Content Processor Module

Handles processing of crawled content for better RAG consumption.
Includes cleaning, filtering, and conversion to markdown.
"""

import re
import json
from typing import Dict, List, Any, Optional
import hashlib
from pathlib import Path
import os
import nltk
from markdownify import markdownify

from config import SETTINGS
from utils import setup_logger

# Set up logger
logger = setup_logger("processor")

# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

class ContentProcessor:
    """
    Process crawled content for better usability in RAG systems.
    """
    
    def __init__(self, chunk_size: int = 1000):
        """
        Initialize the content processor.
        
        Args:
            chunk_size: Target size for content chunks (in characters)
        """
        self.chunk_size = chunk_size
    
    def process(self, crawl_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process crawled data to enhance it for RAG consumption.
        
        Args:
            crawl_data: Raw crawled data from WebCrawler
            
        Returns:
            Processed data with enhanced content
        """
        if not crawl_data or not isinstance(crawl_data, dict) or "pages" not in crawl_data:
            logger.error("Invalid crawl data format")
            return {"error": "Invalid data format", "pages": []}
        
        # Create a new result object
        result = {
            "metadata": crawl_data.get("metadata", {}),
            "pages": []
        }
        
        # Process each page
        for page in crawl_data["pages"]:
            processed_page = self._process_page(page)
            result["pages"].append(processed_page)
        
        # Add additional metadata
        result["metadata"]["processed_date"] = result["metadata"].get("crawl_date", "")
        result["metadata"]["page_count"] = len(result["pages"])
        result["metadata"]["total_word_count"] = sum(
            len(page["content"].split()) for page in result["pages"]
        )
        
        return result
    
    def _process_page(self, page: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single page from crawl results.
        
        Args:
            page: Single page data from crawler
            
        Returns:
            Processed page data
        """
        # Create a copy of the page to avoid modifying the original
        processed = page.copy()
        
        # Clean content if needed
        content = page.get("content", "")
        if content:
            # Clean the content
            content = self._clean_content(content)
            
            # Add summary using nltk
            summary = self._generate_summary(content)
            processed["summary"] = summary
        
        # Make sure content is set
        processed["content"] = content
        
        # Process images if present
        if "images" in page and page["images"]:
            processed["images"] = self._process_images(page["images"])
        
        # Generate a hash ID for the page
        page_id = hashlib.md5(f"{page['url']}".encode()).hexdigest()
        processed["id"] = page_id
        
        return processed
    
    def _clean_content(self, content: str) -> str:
        """
        Clean the content for better RAG usage.
        
        Args:
            content: Raw markdown content
            
        Returns:
            Cleaned content
        """
        # Remove excessive whitespace
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        # Fix markdown formatting issues
        content = re.sub(r'(?<!\n)#{1,6} ', r'\n\1 ', content)
        
        return content
    
    def _generate_summary(self, content: str, max_length: int = 200) -> str:
        """
        Generate a brief summary from content using NLTK.
        
        Args:
            content: Markdown content
            max_length: Maximum summary length in characters
            
        Returns:
            Brief summary of the content
        """
        try:
            # Get first few sentences
            sentences = nltk.sent_tokenize(content)
            
            # Take first 3 sentences as summary
            summary_sentences = sentences[:3]
            summary = ' '.join(summary_sentences)
            
            # Truncate if needed
            if len(summary) > max_length:
                summary = summary[:max_length-3] + "..."
            
            return summary
        
        except Exception as e:
            logger.warning(f"Error generating summary: {e}")
            # Fallback to simpler approach
            return content[:max_length-3] + "..." if len(content) > max_length else content
    
    def _process_images(self, images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process images to handle storage and references.
        
        Args:
            images: List of image data from crawler
            
        Returns:
            Processed image data
        """
        processed_images = []
        
        for img in images:
            # Skip if no source URL
            if not img.get("src"):
                continue
            
            # Copy the image data
            processed_img = img.copy()
            
            # Check if we need to download and store the image
            if SETTINGS.STORE_IMAGES and img.get("src", "").startswith(("http://", "https://")):
                # In a real implementation, download the image here
                # For this prototype, we'll just note that we would download it
                processed_img["local_path"] = f"images/{hashlib.md5(img['src'].encode()).hexdigest()}.jpg"
            
            processed_images.append(processed_img)
        
        return processed_images
    
    def to_markdown(self, processed_data: Dict[str, Any]) -> str:
        """
        Convert processed data to a single markdown document.
        
        Args:
            processed_data: Processed crawl data
            
        Returns:
            Markdown string representation
        """
        if not processed_data or "pages" not in processed_data:
            return "# Error\nNo valid content to convert to markdown."
        
        markdown_parts = []
        
        # Add title
        markdown_parts.append("# Crawled Content")
        
        # Add metadata
        markdown_parts.append("\n## Metadata")
        metadata = processed_data.get("metadata", {})
        for key, value in metadata.items():
            markdown_parts.append(f"- **{key}**: {value}")
        
        # Add pages
        for i, page in enumerate(processed_data["pages"]):
            markdown_parts.append(f"\n## {page.get('title', f'Page {i+1}')}")
            markdown_parts.append(f"**URL**: {page.get('url', 'Unknown')}")
            
            if "summary" in page:
                markdown_parts.append(f"\n### Summary\n{page['summary']}")
            
            markdown_parts.append(f"\n### Content\n{page['content']}")
            
            # Add images if present
            if "images" in page and page["images"]:
                markdown_parts.append("\n### Images")
                for img in page["images"]:
                    alt_text = img.get("alt", "Image")
                    src = img.get("local_path", img.get("src", "#"))
                    markdown_parts.append(f"![{alt_text}]({src})")
        
        return "\n\n".join(markdown_parts)
    
    def to_json(self, processed_data: Dict[str, Any]) -> str:
        """
        Convert processed data to a JSON string.
        
        Args:
            processed_data: Processed crawl data
            
        Returns:
            JSON string representation
        """
        return json.dumps(processed_data, indent=2)
    
    def save_to_file(self, processed_data: Dict[str, Any], output_path: str, 
                     format: str = "markdown") -> str:
        """
        Save processed data to a file.
        
        Args:
            processed_data: Processed crawl data
            output_path: Path to save the file
            format: Format to save as ("markdown" or "json")
            
        Returns:
            Path to the saved file
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if format.lower() == "json":
            content = self.to_json(processed_data)
            if not output_path.endswith(".json"):
                output_path += ".json"
        else:
            content = self.to_markdown(processed_data)
            if not output_path.endswith(".md"):
                output_path += ".md"
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        return output_path
    
    def chunk_content(self, content: str, overlap: int = 100) -> List[str]:
        """
        Split content into chunks for RAG processing.
        
        Args:
            content: Content to chunk
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of content chunks
        """
        if not content:
            return []
        
        chunks = []
        
        # Use NLTK to split into sentences
        sentences = nltk.sent_tokenize(content)
        
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            # If adding this sentence exceeds chunk size, save current chunk and start new one
            if current_size + sentence_size > self.chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                
                # Keep some overlap sentences
                overlap_size = 0
                overlap_sentences = []
                
                # Add sentences from the end of the current chunk for overlap
                for s in reversed(current_chunk):
                    if overlap_size + len(s) <= overlap:
                        overlap_sentences.insert(0, s)
                        overlap_size += len(s)
                    else:
                        break
                
                # Start new chunk with overlap sentences
                current_chunk = overlap_sentences
                current_size = overlap_size
            
            # Add the current sentence
            current_chunk.append(sentence)
            current_size += sentence_size
        
        # Add any remaining content
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks