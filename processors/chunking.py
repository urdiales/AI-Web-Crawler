"""
Chunking Module

This module handles splitting content into appropriate chunks for embedding
and retrieval in a RAG system.
"""

import logging
import re
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# Default chunking parameters
DEFAULT_CHUNK_SIZE = 512  # Target number of tokens
DEFAULT_CHUNK_OVERLAP = 50  # Token overlap between chunks
DEFAULT_MIN_CHUNK_SIZE = 100  # Minimum chunk size to keep

def chunk_content(content: str, 
                 chunk_size: int = DEFAULT_CHUNK_SIZE,
                 chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
                 min_chunk_size: int = DEFAULT_MIN_CHUNK_SIZE,
                 preserve_code_blocks: bool = True) -> List[str]:
    """
    Split content into chunks suitable for embedding.
    
    Args:
        content: Text content to split
        chunk_size: Target size of each chunk (in tokens, approximated by words)
        chunk_overlap: Overlap between chunks (in tokens)
        min_chunk_size: Minimum chunk size to keep
        preserve_code_blocks: Whether to keep code blocks intact
        
    Returns:
        List of chunked text
    """
    try:
        logger.info(f"Chunking content of length {len(content)} into chunks of ~{chunk_size} tokens")
        
        if not content or len(content.strip()) == 0:
            logger.warning("Empty content provided for chunking")
            return []
        
        # First, check if content has code blocks that need preservation
        if preserve_code_blocks:
            chunks = chunk_with_code_preservation(content, chunk_size, chunk_overlap, min_chunk_size)
        else:
            chunks = chunk_by_paragraph(content, chunk_size, chunk_overlap, min_chunk_size)
        
        logger.info(f"Content chunked into {len(chunks)} chunks")
        return chunks
    
    except Exception as e:
        logger.error(f"Error chunking content: {str(e)}", exc_info=True)
        # Fallback to a simple chunking method
        return simple_chunking(content, chunk_size, min_chunk_size)

def chunk_with_code_preservation(content: str, 
                               chunk_size: int,
                               chunk_overlap: int,
                               min_chunk_size: int) -> List[str]:
    """
    Chunk content while preserving code blocks.
    
    Args:
        content: Text content to chunk
        chunk_size: Target chunk size
        chunk_overlap: Overlap size
        min_chunk_size: Minimum chunk size
        
    Returns:
        List of chunks
    """
    # Identify code blocks
    code_blocks = []
    code_block_pattern = r'```(?:[a-zA-Z0-9]+)?\n(.*?)\n```'
    
    # Replace code blocks with placeholders and collect them
    def replace_code_block(match):
        block_id = len(code_blocks)
        code_blocks.append(match.group(0))
        return f"__CODE_BLOCK_{block_id}__"
    
    content_without_code = re.sub(code_block_pattern, replace_code_block, content, flags=re.DOTALL)
    
    # Chunk the content without code blocks
    initial_chunks = chunk_by_paragraph(content_without_code, chunk_size, chunk_overlap, min_chunk_size)
    
    # Restore code blocks
    final_chunks = []
    for chunk in initial_chunks:
        # Find all code block placeholders in this chunk
        placeholders = re.findall(r'__CODE_BLOCK_(\d+)__', chunk)
        
        # If chunk has placeholders, restore them
        if placeholders:
            for block_id in placeholders:
                block_id = int(block_id)
                if block_id < len(code_blocks):
                    # Replace placeholder with original code block
                    chunk = chunk.replace(f"__CODE_BLOCK_{block_id}__", code_blocks[block_id])
        
        final_chunks.append(chunk)
    
    return final_chunks

def chunk_by_paragraph(content: str, 
                     chunk_size: int,
                     chunk_overlap: int,
                     min_chunk_size: int) -> List[str]:
    """
    Chunk content by paragraph boundaries.
    
    Args:
        content: Text content to chunk
        chunk_size: Target chunk size
        chunk_overlap: Overlap size
        min_chunk_size: Minimum chunk size
        
    Returns:
        List of chunks
    """
    # Split by paragraphs (double newlines)
    paragraphs = re.split(r'\n\s*\n', content)
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for paragraph in paragraphs:
        # Skip empty paragraphs
        if not paragraph.strip():
            continue
        
        # Approximate paragraph size in tokens
        paragraph_size = len(paragraph.split())
        
        # If paragraph is larger than chunk_size, split further
        if paragraph_size > chunk_size:
            # If we have content in current_chunk, add it as a chunk
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                # Start next chunk with overlap if possible
                current_chunk = get_overlap_from_chunks(chunks, chunk_overlap)
                current_size = sum(len(part.split()) for part in current_chunk)
            
            # Split large paragraph by sentences
            sentences = split_into_sentences(paragraph)
            sentence_chunks = []
            current_sentence_chunk = []
            current_sentence_size = 0
            
            for sentence in sentences:
                sentence_size = len(sentence.split())
                
                if current_sentence_size + sentence_size <= chunk_size:
                    current_sentence_chunk.append(sentence)
                    current_sentence_size += sentence_size
                else:
                    if current_sentence_chunk:
                        sentence_chunks.append(' '.join(current_sentence_chunk))
                    
                    # If sentence itself is too large, split it
                    if sentence_size > chunk_size:
                        words = sentence.split()
                        for i in range(0, len(words), chunk_size):
                            part = ' '.join(words[i:i+chunk_size])
                            if len(part.split()) >= min_chunk_size:
                                sentence_chunks.append(part)
                        current_sentence_chunk = []
                        current_sentence_size = 0
                    else:
                        current_sentence_chunk = [sentence]
                        current_sentence_size = sentence_size
            
            if current_sentence_chunk:
                sentence_chunks.append(' '.join(current_sentence_chunk))
            
            # Add these sentence chunks to our final chunks
            chunks.extend(sentence_chunks)
            
            # Start fresh with next paragraph
            current_chunk = []
            current_size = 0
            
        # Normal case: add paragraph to current chunk if it fits
        elif current_size + paragraph_size <= chunk_size:
            current_chunk.append(paragraph)
            current_size += paragraph_size
        else:
            # Chunk is full, start a new one
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
            
            # Start next chunk with overlap if possible
            current_chunk = get_overlap_from_chunks(chunks, chunk_overlap)
            current_chunk.append(paragraph)
            current_size = sum(len(part.split()) for part in current_chunk)
    
    # Add the last chunk if it's non-empty and meets minimum size
    if current_chunk and current_size >= min_chunk_size:
        chunks.append('\n\n'.join(current_chunk))
    
    return chunks

def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences.
    
    Args:
        text: Text to split
        
    Returns:
        List of sentences
    """
    # Basic sentence splitting
    sentence_endings = r'(?<=[.!?])\s+'
    sentences = re.split(sentence_endings, text)
    
    # Filter out empty sentences
    return [s for s in sentences if s.strip()]

def get_overlap_from_chunks(chunks: List[str], overlap_size: int) -> List[str]:
    """
    Get overlapping content from the previous chunk.
    
    Args:
        chunks: List of existing chunks
        overlap_size: Desired overlap size
        
    Returns:
        List of paragraph parts for the overlap
    """
    if not chunks:
        return []
    
    last_chunk = chunks[-1]
    
    # Split the last chunk into paragraphs
    paragraphs = last_chunk.split('\n\n')
    
    # Count words in each paragraph from the end
    words_so_far = 0
    overlap_paragraphs = []
    
    for paragraph in reversed(paragraphs):
        paragraph_size = len(paragraph.split())
        
        if words_so_far + paragraph_size <= overlap_size:
            overlap_paragraphs.insert(0, paragraph)
            words_so_far += paragraph_size
        else:
            # If a single paragraph is too large, get the last N words
            if not overlap_paragraphs:
                words = paragraph.split()
                if len(words) > overlap_size:
                    overlap_text = ' '.join(words[-overlap_size:])
                    overlap_paragraphs.insert(0, overlap_text)
            break
    
    return overlap_paragraphs

def simple_chunking(content: str, chunk_size: int, min_chunk_size: int) -> List[str]:
    """
    Simple fallback chunking method that splits by words.
    
    Args:
        content: Text content
        chunk_size: Target chunk size
        min_chunk_size: Minimum chunk size
        
    Returns:
        List of chunks
    """
    words = content.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i+chunk_size])
        if len(chunk.split()) >= min_chunk_size:
            chunks.append(chunk)
    
    return chunks

def chunk_markdown_preserving_headers(markdown: str, 
                                    chunk_size: int = DEFAULT_CHUNK_SIZE,
                                    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[Dict[str, Any]]:
    """
    Chunk markdown content while preserving header context.
    
    Args:
        markdown: Markdown content
        chunk_size: Target chunk size
        chunk_overlap: Overlap size
        
    Returns:
        List of dictionaries with chunk content and contextual information
    """
    try:
        # Split markdown by headers
        header_pattern = r'^(#{1,6}\s+.*?)$'
        sections = re.split(header_pattern, markdown, flags=re.MULTILINE)
        
        # Process sections into header-content pairs
        header_content_pairs = []
        current_headers = []
        
        i = 0
        while i < len(sections):
            if i+1 < len(sections) and re.match(header_pattern, sections[i], re.MULTILINE):
                # This is a header followed by content
                header = sections[i].strip()
                content = sections[i+1].strip()
                
                # Update current headers based on header level
                header_level = len(re.match(r'^(#+)', header).group(1))
                
                # Remove headers of same or lower levels
                current_headers = [h for h in current_headers if get_header_level(h) < header_level]
                current_headers.append(header)
                
                header_content_pairs.append({
                    "headers": current_headers.copy(),
                    "content": content
                })
                
                i += 2
            else:
                # This is content without a preceding header
                content = sections[i].strip()
                if content:
                    header_content_pairs.append({
                        "headers": current_headers.copy(),
                        "content": content
                    })
                i += 1
        
        # Chunk the content while preserving header context
        chunks = []
        
        for pair in header_content_pairs:
            headers = pair["headers"]
            content = pair["content"]
            
            # Skip if content is too small
            if not content or len(content.split()) < 10:
                continue
            
            # Chunk the content
            content_chunks = chunk_content(content, chunk_size, chunk_overlap)
            
            for chunk in content_chunks:
                # Create header prefix with context
                header_prefix = ""
                for h in headers:
                    # Only include the header text, not the markdown #s
                    header_text = re.sub(r'^#+\s+', '', h).strip()
                    header_prefix += f"{header_text} > "
                
                header_prefix = header_prefix.rstrip(" >")
                
                chunks.append({
                    "content": chunk,
                    "context": header_prefix,
                    "headers": headers
                })
        
        return chunks
    
    except Exception as e:
        logger.error(f"Error chunking markdown with headers: {str(e)}", exc_info=True)
        # Fallback to simple chunking
        simple_chunks = chunk_content(markdown, chunk_size, chunk_overlap)
        return [{"content": chunk, "context": "", "headers": []} for chunk in simple_chunks]

def get_header_level(header: str) -> int:
    """
    Get the level of a markdown header.
    
    Args:
        header: Markdown header string
        
    Returns:
        Header level (1-6)
    """
    match = re.match(r'^(#+)', header)
    if match:
        return len(match.group(1))
    return 0