"""
Image Handler

This module handles downloading, processing, and storing images from crawled web pages.
"""

import asyncio
import aiohttp
import logging
import os
from pathlib import Path
import time
import hashlib
from typing import List, Dict, Any
from urllib.parse import urlparse, urljoin
import mimetypes
from PIL import Image
import io

logger = logging.getLogger(__name__)

# Define image storage directory
IMAGE_DIR = Path("data/images")
IMAGE_DIR.mkdir(exist_ok=True, parents=True)

async def process_images(images: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Process and download images from crawled pages.
    
    Args:
        images: List of image data dictionaries
        
    Returns:
        Dict with processed image information
    """
    logger.info(f"Processing {len(images)} images")
    start_time = time.time()
    
    # Filter out images without source
    valid_images = [img for img in images if 'src' in img and img['src']]
    
    if not valid_images:
        logger.info("No valid images found to process")
        return {
            "success": True,
            "processed_images": [],
            "duration_seconds": time.time() - start_time
        }
    
    # Create a subdirectory for this batch
    batch_id = int(time.time())
    batch_dir = IMAGE_DIR / f"batch_{batch_id}"
    batch_dir.mkdir(exist_ok=True)
    
    # Process images concurrently
    tasks = [download_and_save_image(img, batch_dir) for img in valid_images]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out errors and collect successful downloads
    processed_images = []
    errors = []
    
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.warning(f"Error processing image: {str(result)}")
            errors.append({
                "src": valid_images[i].get('src', 'unknown'),
                "error": str(result)
            })
        elif result:
            processed_images.append(result)
    
    duration = time.time() - start_time
    logger.info(f"Image processing completed in {duration:.2f} seconds. "
                f"Downloaded {len(processed_images)} of {len(valid_images)} images.")
    
    return {
        "success": True,
        "processed_images": processed_images,
        "errors": errors,
        "duration_seconds": duration,
        "batch_dir": str(batch_dir)
    }

async def download_and_save_image(image_data: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    """
    Download and save a single image.
    
    Args:
        image_data: Dictionary with image information
        output_dir: Directory to save the image
        
    Returns:
        Updated image data dictionary with local file information
    """
    try:
        src = image_data.get('src', '')
        if not src:
            return None
        
        # Ensure absolute URL
        page_url = image_data.get('page_url', '')
        if page_url and not src.startswith(('http://', 'https://')):
            src = urljoin(page_url, src)
        
        # Create a unique filename based on URL
        url_hash = hashlib.md5(src.encode('utf-8')).hexdigest()
        
        # Try to determine file extension from URL or content type
        parsed_url = urlparse(src)
        path = parsed_url.path
        ext = os.path.splitext(path)[1].lower()
        
        if not ext or ext == '.':
            # Default to .jpg if we can't determine the extension
            ext = '.jpg'
        
        filename = f"{url_hash}{ext}"
        file_path = output_dir / filename
        
        # Download the image
        async with aiohttp.ClientSession() as session:
            async with session.get(src, timeout=30) as response:
                if response.status != 200:
                    raise Exception(f"HTTP error {response.status} for {src}")
                
                # Get content type and verify it's an image
                content_type = response.headers.get('Content-Type', '')
                if not content_type.startswith('image/'):
                    raise Exception(f"Not an image: {content_type}")
                
                image_data = await response.read()
                
                # Attempt to open with PIL to verify it's a valid image
                try:
                    img = Image.open(io.BytesIO(image_data))
                    width, height = img.size
                    format = img.format.lower() if img.format else 'unknown'
                except Exception as e:
                    raise Exception(f"Invalid image data: {str(e)}")
                
                # Save the image
                with open(file_path, 'wb') as f:
                    f.write(image_data)
                
                # Return updated image data
                result = {
                    **image_data,
                    "local_path": str(file_path),
                    "filename": filename,
                    "width": width,
                    "height": height,
                    "format": format,
                    "size_bytes": len(image_data)
                }
                
                return result
    
    except Exception as e:
        logger.warning(f"Error downloading image {image_data.get('src', 'unknown')}: {str(e)}")
        raise e

def get_image_thumbnail(image_path: str, max_size: int = 100) -> str:
    """
    Generate a thumbnail for an image.
    
    Args:
        image_path: Path to the image file
        max_size: Maximum dimension (width or height)
        
    Returns:
        Base64-encoded string of the thumbnail
    """
    try:
        import base64
        
        img = Image.open(image_path)
        
        # Calculate thumbnail size while preserving aspect ratio
        img.thumbnail((max_size, max_size))
        
        # Save to a bytes buffer
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return f"data:image/jpeg;base64,{img_str}"
    
    except Exception as e:
        logger.warning(f"Error generating thumbnail: {str(e)}")
        return ""