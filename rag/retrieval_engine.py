"""
Retrieval Engine

This module handles the retrieval of relevant context for RAG queries.
"""

import logging
import os
from typing import List, Dict, Any, Optional, Tuple
import time
import json
from pathlib import Path
import asyncio

from database.embeddings import generate_embeddings
from database.vector_store import search_vectors, get_vector_store_info
from processors.markdown_generator import generate_markdown

logger = logging.getLogger(__name__)

class RetrievalEngine:
    """Handles the retrieval of relevant context for RAG queries."""
    
    def __init__(self, index_name: str = "default"):
        """
        Initialize the retrieval engine.
        
        Args:
            index_name: Name of the vector index to use
        """
        self.index_name = index_name
    
    async def retrieve(self, 
                      query: str, 
                      top_k: int = 5,
                      min_score: float = 0.0) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Retrieve relevant context for a query.
        
        Args:
            query: User query
            top_k: Number of results to retrieve
            min_score: Minimum similarity score threshold
            
        Returns:
            Tuple of (results, context strings)
        """
        try:
            logger.info(f"Retrieving context for query: '{query}'")
            
            # Generate embeddings for the query
            embeddings = await generate_embeddings([query])
            if not embeddings or len(embeddings) == 0:
                logger.warning("Failed to generate query embeddings")
                return [], []
            
            query_embedding = embeddings[0]
            
            # Search for similar vectors
            results = await search_vectors(
                query_embedding=query_embedding,
                top_k=top_k,
                index_name=self.index_name
            )
            
            # Filter by minimum score if needed
            if min_score > 0:
                results = [r for r in results if r.get("score", 0) >= min_score]
            
            # Extract context strings
            context = [result["content"] for result in results]
            
            logger.info(f"Retrieved {len(context)} context chunks")
            return results, context
        
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}", exc_info=True)
            return [], []
    
    async def retrieve_from_json(self, 
                               query: str,
                               json_path: str,
                               top_k: int = 5) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Retrieve relevant context from a JSON file without using embeddings.
        
        Args:
            query: User query
            json_path: Path to the JSON file
            top_k: Number of results to retrieve
            
        Returns:
            Tuple of (results, context strings)
        """
        try:
            logger.info(f"Retrieving context from JSON file: {json_path}")
            
            # Read JSON file
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract pages
            pages = data.get('pages', [])
            if not pages:
                logger.warning(f"No pages found in JSON file: {json_path}")
                return [], []
            
            # Simple keyword-based search (fallback when vector search is not available)
            query_terms = query.lower().split()
            
            scored_pages = []
            for page in pages:
                # Get page content (concatenate all sections)
                content = ""
                for section in page.get('content', {}).get('sections', []):
                    content += section.get('content', '') + " "
                
                # Count query term occurrences
                score = 0
                for term in query_terms:
                    if term and len(term) > 2:  # Skip very short terms
                        score += content.lower().count(term)
                
                if score > 0:
                    scored_pages.append({
                        "page": page,
                        "score": score,
                        "content": content
                    })
            
            # Sort by score and take top_k
            scored_pages.sort(key=lambda x: x["score"], reverse=True)
            top_results = scored_pages[:top_k]
            
            # Format results
            results = []
            context = []
            
            for item in top_results:
                page = item["page"]
                url = page.get('url', '')
                title = page.get('title', 'Untitled')
                
                results.append({
                    "url": url,
                    "title": title,
                    "content": item["content"],
                    "score": item["score"]
                })
                
                context.append(f"## {title}\n\n{item['content']}")
            
            logger.info(f"Retrieved {len(context)} context chunks from JSON")
            return results, context
        
        except Exception as e:
            logger.error(f"Error retrieving from JSON: {str(e)}", exc_info=True)
            return [], []
    
    async def retrieve_from_markdown_files(self, 
                                         query: str,
                                         markdown_dir: str,
                                         top_k: int = 5) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Retrieve relevant context from Markdown files without using embeddings.
        
        Args:
            query: User query
            markdown_dir: Directory containing Markdown files
            top_k: Number of results to retrieve
            
        Returns:
            Tuple of (results, context strings)
        """
        try:
            logger.info(f"Retrieving context from Markdown files in: {markdown_dir}")
            
            # Get all markdown files
            markdown_path = Path(markdown_dir)
            if not markdown_path.exists() or not markdown_path.is_dir():
                logger.warning(f"Markdown directory does not exist: {markdown_dir}")
                return [], []
            
            markdown_files = list(markdown_path.glob("**/*.md"))
            if not markdown_files:
                logger.warning(f"No Markdown files found in: {markdown_dir}")
                return [], []
            
            # Read all markdown files
            file_contents = []
            for file_path in markdown_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                        # Extract title from first header or filename
                        title = os.path.basename(file_path).replace('.md', '')
                        header_match = content.split('\n', 1)[0] if content else ''
                        if header_match.startswith('# '):
                            title = header_match[2:].strip()
                        
                        file_contents.append({
                            "path": str(file_path),
                            "title": title,
                            "content": content
                        })
                except Exception as e:
                    logger.warning(f"Error reading markdown file {file_path}: {str(e)}")
            
            # Simple keyword-based search
            query_terms = query.lower().split()
            
            scored_files = []
            for file_data in file_contents:
                content = file_data["content"].lower()
                
                # Count query term occurrences
                score = 0
                for term in query_terms:
                    if term and len(term) > 2:  # Skip very short terms
                        score += content.count(term)
                
                if score > 0:
                    scored_files.append({
                        "file": file_data,
                        "score": score
                    })
            
            # Sort by score and take top_k
            scored_files.sort(key=lambda x: x["score"], reverse=True)
            top_results = scored_files[:top_k]
            
            # Format results
            results = []
            context = []
            
            for item in top_results:
                file_data = item["file"]
                
                results.append({
                    "path": file_data["path"],
                    "title": file_data["title"],
                    "content": file_data["content"],
                    "score": item["score"]
                })
                
                context.append(file_data["content"])
            
            logger.info(f"Retrieved {len(context)} context chunks from Markdown files")
            return results, context
        
        except Exception as e:
            logger.error(f"Error retrieving from Markdown: {str(e)}", exc_info=True)
            return [], []
    
    async def get_recent_crawls(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get information about recent crawls.
        
        Args:
            limit: Maximum number of crawls to return
            
        Returns:
            List of crawl information dictionaries
        """
        try:
            # Check JSON directory for crawl files
            json_dir = Path("data/json")
            if not json_dir.exists() or not json_dir.is_dir():
                return []
            
            # Get all JSON files
            json_files = list(json_dir.glob("*.json"))
            if not json_files:
                return []
            
            # Sort by modification time (most recent first)
            json_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            
            # Get information from each file
            crawls = []
            for file_path in json_files[:limit]:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Extract basic information
                    crawl_info = {
                        "id": data.get("crawl_id", os.path.basename(file_path).replace('.json', '')),
                        "base_url": data.get("base_url", "Unknown"),
                        "domain": data.get("domain", "Unknown"),
                        "date": data.get("date", "Unknown"),
                        "page_count": data.get("page_count", 0),
                        "file_path": str(file_path),
                        "file_size": os.path.getsize(file_path)
                    }
                    
                    crawls.append(crawl_info)
                except Exception as e:
                    logger.warning(f"Error reading JSON file {file_path}: {str(e)}")
            
            return crawls
        
        except Exception as e:
            logger.error(f"Error getting recent crawls: {str(e)}", exc_info=True)
            return []
                    "