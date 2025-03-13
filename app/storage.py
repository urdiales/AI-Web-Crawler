"""
Storage Module

Handles persistent storage of crawled content and knowledge base management.
Includes vector storage for semantic search capabilities.
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import hashlib
import uuid

import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

from config import SETTINGS
from utils import setup_logger

# Set up logger
logger = setup_logger("storage")

class KnowledgeBase:
    """
    Manages persistent storage and retrieval of crawled content.
    """
    
    def __init__(self, knowledge_base_dir: str):
        """
        Initialize the knowledge base.
        
        Args:
            knowledge_base_dir: Directory for storing knowledge base data
        """
        self.knowledge_base_dir = Path(knowledge_base_dir)
        self.knowledge_base_dir.mkdir(parents=True, exist_ok=True)
        
        # Create vector database directory
        self.vector_db_path = self.knowledge_base_dir / "vector_db"
        self.vector_db_path.mkdir(exist_ok=True)
        
        # Initialize ChromaDB for vector storage
        self.chroma_client = chromadb.PersistentClient(path=str(self.vector_db_path))
        
        # Initialize sentence transformer for embeddings if configured
        self.sentence_transformer = None
        self.embedding_function = None
        
        if SETTINGS.USE_EMBEDDINGS:
            try:
                # Try to load the embedding model
                self._initialize_embeddings()
            except Exception as e:
                logger.error(f"Error initializing embeddings: {e}")
    
    def _initialize_embeddings(self):
        """Initialize the embedding model for vector search."""
        try:
            # If we're using a local embedding model
            if SETTINGS.USE_LOCAL_EMBEDDINGS:
                self.sentence_transformer = SentenceTransformer(SETTINGS.EMBEDDING_MODEL)
                
                # Create a custom embedding function using the model
                def embed_function(texts):
                    return self.sentence_transformer.encode(texts).tolist()
                
                self.embedding_function = embed_function
            
            # Otherwise use the default from chromadb
            else:
                # Use OpenAI or HuggingFace embeddings
                if "openai" in SETTINGS.EMBEDDING_MODEL.lower():
                    self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                        api_key=SETTINGS.OPENAI_API_KEY,
                        model_name=SETTINGS.EMBEDDING_MODEL
                    )
                else:
                    self.embedding_function = embedding_functions.HuggingFaceEmbeddingFunction(
                        model_name=SETTINGS.EMBEDDING_MODEL
                    )
            
            logger.info(f"Initialized embeddings with model: {SETTINGS.EMBEDDING_MODEL}")
        
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            # Fall back to default text matching
            self.embedding_function = None
    
    def add_content(self, content_data: Dict[str, Any], name: Optional[str] = None) -> str:
        """
        Add crawled content to the knowledge base.
        
        Args:
            content_data: Processed crawl data to add
            name: Optional name for the knowledge base (defaults to timestamp)
            
        Returns:
            ID of the created knowledge base entry
        """
        # Generate ID and name if not provided
        kb_id = str(uuid.uuid4())
        
        if not name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"crawl_{timestamp}"
        
        # Create metadata for the knowledge base
        kb_info = {
            "id": kb_id,
            "name": name,
            "created": datetime.now().isoformat(),
            "page_count": len(content_data.get("pages", [])),
            "source": content_data.get("metadata", {}).get("url", "unknown"),
            "embedding_model": SETTINGS.EMBEDDING_MODEL if SETTINGS.USE_EMBEDDINGS else None
        }
        
        # Create knowledge base directory
        kb_dir = self.knowledge_base_dir / kb_id
        kb_dir.mkdir(exist_ok=True)
        
        # Save metadata
        with open(kb_dir / "info.json", "w") as f:
            json.dump(kb_info, f, indent=2)
        
        # Save content data
        with open(kb_dir / "content.json", "w") as f:
            json.dump(content_data, f, indent=2)
        
        # Create vector store if enabled
        if SETTINGS.USE_EMBEDDINGS and self.embedding_function:
            try:
                self._create_vector_index(kb_id, content_data)
                logger.info(f"Created vector index for KB: {kb_id}")
            except Exception as e:
                logger.error(f"Error creating vector index: {e}")
        
        logger.info(f"Added content to knowledge base: {kb_id}")
        return kb_id
    
    def _create_vector_index(self, kb_id: str, content_data: Dict[str, Any]) -> None:
        """
        Create vector embeddings for the content.
        
        Args:
            kb_id: Knowledge base ID
            content_data: Content data to index
        """
        if not self.embedding_function:
            logger.warning("Embeddings not initialized, skipping vector indexing")
            return
        
        # Create or get collection
        try:
            collection = self.chroma_client.get_or_create_collection(
                name=kb_id,
                embedding_function=self.embedding_function
            )
            
            # Extract chunks from pages
            documents = []
            metadatas = []
            ids = []
            
            # Process each page
            for page in content_data.get("pages", []):
                page_content = page.get("content", "")
                if not page_content:
                    continue
                
                # Use page content as a single chunk for simplicity
                # In a production system, you'd want to chunk the content properly
                chunk_id = f"{kb_id}_{hashlib.md5(page['url'].encode()).hexdigest()}"
                
                documents.append(page_content)
                metadatas.append({
                    "url": page.get("url", ""),
                    "title": page.get("title", ""),
                    "source": kb_id
                })
                ids.append(chunk_id)
            
            # Add documents to the collection
            if documents:
                collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                
                logger.info(f"Added {len(documents)} documents to vector index for KB: {kb_id}")
            else:
                logger.warning(f"No content to add to vector index for KB: {kb_id}")
        
        except Exception as e:
            logger.error(f"Error in vector indexing: {e}")
            raise
    
    def get_knowledge_base(self, kb_id: str) -> Dict[str, Any]:
        """
        Get knowledge base content by ID.
        
        Args:
            kb_id: Knowledge base ID
            
        Returns:
            Knowledge base content
        """
        kb_dir = self.knowledge_base_dir / kb_id
        
        if not kb_dir.exists():
            logger.error(f"Knowledge base not found: {kb_id}")
            return {"error": "Knowledge base not found"}
        
        # Load info
        try:
            with open(kb_dir / "info.json", "r") as f:
                kb_info = json.load(f)
            
            # Load content
            with open(kb_dir / "content.json", "r") as f:
                content_data = json.load(f)
            
            # Combine info and content
            result = {**kb_info, **content_data}
            return result
            
        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}")
            return {"error": f"Error loading knowledge base: {str(e)}"}
    
    def delete_knowledge_base(self, kb_id: str) -> bool:
        """
        Delete a knowledge base.
        
        Args:
            kb_id: Knowledge base ID
            
        Returns:
            Whether deletion was successful
        """
        kb_dir = self.knowledge_base_dir / kb_id
        
        if not kb_dir.exists():
            logger.error(f"Knowledge base not found: {kb_id}")
            return False
        
        try:
            # Delete the directory
            shutil.rmtree(kb_dir)
            
            # Delete vector collection if it exists
            try:
                self.chroma_client.delete_collection(kb_id)
            except Exception as e:
                logger.warning(f"Error deleting vector collection: {e}")
            
            logger.info(f"Deleted knowledge base: {kb_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting knowledge base: {e}")
            return False
    
    def list_knowledge_bases(self) -> Dict[str, Dict[str, Any]]:
        """
        List all knowledge bases.
        
        Returns:
            Dict mapping knowledge base IDs to their metadata
        """
        result = {}
        
        # Find all directories in the knowledge base directory
        for item in self.knowledge_base_dir.iterdir():
            if item.is_dir() and (item / "info.json").exists():
                try:
                    with open(item / "info.json", "r") as f:
                        kb_info = json.load(f)
                    
                    kb_id = item.name
                    result[kb_id] = kb_info
                    
                except Exception as e:
                    logger.warning(f"Error loading knowledge base info: {e}")
        
        return result
    
    def export_knowledge_base(self, kb_id: str, format: str = "json") -> str:
        """
        Export a knowledge base to a file.
        
        Args:
            kb_id: Knowledge base ID
            format: Export format ("json" or "markdown")
            
        Returns:
            Path to the exported file
        """
        kb_data = self.get_knowledge_base(kb_id)
        
        if "error" in kb_data:
            logger.error(f"Error exporting knowledge base: {kb_data['error']}")
            return ""
        
        # Create exports directory
        exports_dir = Path(SETTINGS.EXPORTS_DIRECTORY)
        exports_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        kb_name = kb_data.get("name", kb_id)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format.lower() == "json":
            filename = f"{kb_name}_{timestamp}.json"
            output_path = exports_dir / filename
            
            with open(output_path, "w") as f:
                json.dump(kb_data, f, indent=2)
                
        elif format.lower() == "markdown":
            filename = f"{kb_name}_{timestamp}.md"
            output_path = exports_dir / filename
            
            # Create markdown content
            md_parts = []
            
            # Add title and metadata
            md_parts.append(f"# {kb_data.get('name', 'Knowledge Base')}")
            md_parts.append(f"Created: {kb_data.get('created', '')}")
            md_parts.append(f"Pages: {kb_data.get('page_count', 0)}")
            md_parts.append(f"Source: {kb_data.get('source', '')}")
            
            # Add pages
            for i, page in enumerate(kb_data.get("pages", [])):
                md_parts.append(f"\n## {page.get('title', f'Page {i+1}')}")
                md_parts.append(f"URL: {page.get('url', '')}")
                
                if "summary" in page:
                    md_parts.append(f"\n### Summary\n{page['summary']}")
                
                md_parts.append(f"\n### Content\n{page.get('content', '')}")
                
                # Add images if present
                if "images" in page and page["images"]:
                    md_parts.append("\n### Images")
                    for img in page["images"]:
                        alt = img.get("alt", "Image")
                        src = img.get("src", "")
                        md_parts.append(f"![{alt}]({src})")
            
            # Write to file
            with open(output_path, "w") as f:
                f.write("\n\n".join(md_parts))
        
        else:
            logger.error(f"Unsupported export format: {format}")
            return ""
        
        logger.info(f"Exported knowledge base to: {output_path}")
        return str(output_path)
    
    def search(self, query: str, kb_id: Optional[str] = None, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search the knowledge base for relevant content.
        
        Args:
            query: Search query
            kb_id: Optional knowledge base ID to search within
            limit: Maximum number of results
            
        Returns:
            List of relevant content items
        """
        if not query:
            return []
        
        # If embeddings are enabled, use vector search
        if SETTINGS.USE_EMBEDDINGS and self.embedding_function:
            return self._vector_search(query, kb_id, limit)
        else:
            # Fall back to simple text search
            return self._text_search(query, kb_id, limit)
    
    def _vector_search(self, query: str, kb_id: Optional[str] = None, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Perform vector-based semantic search.
        
        Args:
            query: Search query
            kb_id: Optional knowledge base ID to search within
            limit: Maximum number of results
            
        Returns:
            List of relevant content items
        """
        results = []
        
        try:
            # If kb_id is specified, search only that collection
            if kb_id:
                try:
                    collection = self.chroma_client.get_collection(name=kb_id)
                    search_results = collection.query(
                        query_texts=[query],
                        n_results=limit
                    )
                    
                    # Process results
                    if search_results and len(search_results["documents"]) > 0:
                        documents = search_results["documents"][0]
                        metadatas = search_results["metadatas"][0]
                        distances = search_results.get("distances", [[0] * len(documents)])[0]
                        
                        for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
                            results.append({
                                "content": doc,
                                "url": meta.get("url", ""),
                                "title": meta.get("title", ""),
                                "score": 1.0 - (dist / 2.0) if dist else 1.0  # Convert distance to score
                            })
                
                except Exception as e:
                    logger.error(f"Error searching collection {kb_id}: {e}")
            
            # Otherwise search all collections
            else:
                for collection_name in self.chroma_client.list_collections():
                    try:
                        collection = self.chroma_client.get_collection(name=collection_name.name)
                        search_results = collection.query(
                            query_texts=[query],
                            n_results=min(limit, 3)  # Limit per collection
                        )
                        
                        # Process results
                        if search_results and len(search_results["documents"]) > 0:
                            documents = search_results["documents"][0]
                            metadatas = search_results["metadatas"][0]
                            distances = search_results.get("distances", [[0] * len(documents)])[0]
                            
                            for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
                                results.append({
                                    "content": doc,
                                    "url": meta.get("url", ""),
                                    "title": meta.get("title", ""),
                                    "score": 1.0 - (dist / 2.0) if dist else 1.0,  # Convert distance to score
                                    "kb_id": collection_name.name
                                })
                    
                    except Exception as e:
                        logger.error(f"Error searching collection {collection_name}: {e}")
                
                # Sort by score
                results.sort(key=lambda x: x.get("score", 0), reverse=True)
                
                # Limit total results
                results = results[:limit]
        
        except Exception as e:
            logger.error(f"Error in vector search: {e}")
        
        return results
    
    def _text_search(self, query: str, kb_id: Optional[str] = None, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Perform simple text-based search.
        
        Args:
            query: Search query
            kb_id: Optional knowledge base ID to search within
            limit: Maximum number of results
            
        Returns:
            List of relevant content items
        """
        results = []
        query_terms = query.lower().split()
        
        # Helper function to calculate simple relevance score
        def calculate_score(content, title):
            score = 0
            content_lower = content.lower()
            title_lower = title.lower()
            
            # Check for exact phrase match (highest score)
            if query.lower() in content_lower:
                score += 10
            if query.lower() in title_lower:
                score += 15
            
            # Check for term matches
            for term in query_terms:
                if term in content_lower:
                    score += 1
                if term in title_lower:
                    score += 2
            
            return score
        
        # Get knowledge bases to search
        kb_list = {}
        if kb_id:
            kb_data = self.get_knowledge_base(kb_id)
            if "error" not in kb_data:
                kb_list[kb_id] = kb_data
        else:
            kb_list = self.list_knowledge_bases()
        
        # Search each knowledge base
        for current_kb_id, kb_info in kb_list.items():
            try:
                kb_data = kb_info if "pages" in kb_info else self.get_knowledge_base(current_kb_id)
                
                if "error" in kb_data:
                    continue
                
                for page in kb_data.get("pages", []):
                    content = page.get("content", "")
                    title = page.get("title", "")
                    
                    # Calculate relevance score
                    score = calculate_score(content, title)
                    
                    if score > 0:
                        results.append({
                            "content": content,
                            "url": page.get("url", ""),
                            "title": title,
                            "score": score,
                            "kb_id": current_kb_id
                        })
            
            except Exception as e:
                logger.error(f"Error searching knowledge base {current_kb_id}: {e}")
        
        # Sort by score and limit results
        results.sort(key=lambda x: x.get("score", 0), reverse=True)
        return results[:limit]