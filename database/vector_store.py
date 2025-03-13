"""
Vector Store

This module handles the storage and retrieval of vector embeddings for the RAG system.
"""

import logging
import os
import json
from typing import List, Dict, Any, Optional, Tuple
import time
import numpy as np
from pathlib import Path
import faiss
import pickle

logger = logging.getLogger(__name__)

# Define vector storage directory
VECTOR_DIR = Path("data/vector_db")
VECTOR_DIR.mkdir(exist_ok=True, parents=True)

class VectorStore:
    """Simple vector store using FAISS for efficient similarity search."""
    
    def __init__(self, index_name: str = "default"):
        """
        Initialize the vector store.
        
        Args:
            index_name: Name of the vector index
        """
        self.index_name = index_name
        self.index_path = VECTOR_DIR / f"{index_name}.index"
        self.metadata_path = VECTOR_DIR / f"{index_name}.metadata"
        self.embeddings_path = VECTOR_DIR / f"{index_name}.embeddings"
        
        # Initialize index and metadata
        self.index = None
        self.metadata = {}
        self.embeddings = []
        
        # Load existing index if available
        self._load_index()
    
    def _load_index(self) -> None:
        """Load existing index and metadata if available."""
        try:
            if self.index_path.exists() and self.metadata_path.exists():
                self.index = faiss.read_index(str(self.index_path))
                
                with open(self.metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)
                
                with open(self.embeddings_path, 'rb') as f:
                    self.embeddings = pickle.load(f)
                
                logger.info(f"Loaded existing index {self.index_name} with {self.index.ntotal} vectors")
            else:
                logger.info(f"No existing index found for {self.index_name}, creating new index")
                self.metadata = {
                    "urls": [],
                    "contents": [],
                    "created": time.time(),
                    "updated": time.time(),
                    "count": 0
                }
                self.embeddings = []
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}", exc_info=True)
            # Start with a fresh index
            self.metadata = {
                "urls": [],
                "contents": [],
                "created": time.time(),
                "updated": time.time(),
                "count": 0
            }
            self.embeddings = []
    
    def _initialize_index(self, dim: int) -> None:
        """
        Initialize a FAISS index.
        
        Args:
            dim: Dimension of the vectors
        """
        try:
            # Create a new index
            self.index = faiss.IndexFlatL2(dim)
            
            logger.info(f"Initialized new index with dimension {dim}")
        except Exception as e:
            logger.error(f"Error initializing index: {str(e)}", exc_info=True)
            raise e
    
    def add_embeddings(self, 
                      embeddings: List[List[float]], 
                      urls: List[str], 
                      contents: List[str]) -> bool:
        """
        Add embeddings to the vector store.
        
        Args:
            embeddings: List of embedding vectors
            urls: List of URLs corresponding to the embeddings
            contents: List of text contents corresponding to the embeddings
        
        Returns:
            Boolean indicating success
        """
        try:
            if not embeddings or len(embeddings) == 0:
                logger.warning("No embeddings provided to add")
                return False
            
            # Convert to numpy array
            vectors = np.array(embeddings).astype('float32')
            
            # Initialize index if needed
            if self.index is None:
                self._initialize_index(vectors.shape[1])
            
            # Add vectors to index
            self.index.add(vectors)
            
            # Update metadata
            start_idx = len(self.metadata["urls"])
            self.metadata["urls"].extend(urls)
            self.metadata["contents"].extend(contents)
            self.metadata["updated"] = time.time()
            self.metadata["count"] += len(embeddings)
            
            # Store embeddings
            self.embeddings.extend(embeddings)
            
            # Save index and metadata
            self._save_index()
            
            logger.info(f"Added {len(embeddings)} vectors to index {self.index_name}")
            return True
        
        except Exception as e:
            logger.error(f"Error adding embeddings: {str(e)}", exc_info=True)
            return False
    
    def _save_index(self) -> None:
        """Save the index and metadata to disk."""
        try:
            if self.index is None:
                logger.warning("No index to save")
                return
            
            # Save index
            faiss.write_index(self.index, str(self.index_path))
            
            # Save metadata
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)
            
            # Save embeddings
            with open(self.embeddings_path, 'wb') as f:
                pickle.dump(self.embeddings, f)
            
            logger.info(f"Saved index {self.index_name} with {self.index.ntotal} vectors")
        
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}", exc_info=True)
    
    def search(self, 
              query_embedding: List[float], 
              top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            
        Returns:
            List of dictionaries with search results
        """
        try:
            if self.index is None or self.index.ntotal == 0:
                logger.warning("No index available for search")
                return []
            
            # Convert to numpy array
            query_vector = np.array([query_embedding]).astype('float32')
            
            # Search index
            distances, indices = self.index.search(query_vector, min(top_k, self.index.ntotal))
            
            # Format results
            results = []
            for i, idx in enumerate(indices[0]):
                if idx != -1:  # FAISS returns -1 for padded results
                    results.append({
                        "url": self.metadata["urls"][idx],
                        "content": self.metadata["contents"][idx],
                        "distance": float(distances[0][i]),
                        "score": 1.0 / (1.0 + float(distances[0][i]))  # Convert distance to similarity score
                    })
            
            logger.info(f"Search returned {len(results)} results")
            return results
        
        except Exception as e:
            logger.error(f"Error searching index: {str(e)}", exc_info=True)
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with statistics
        """
        try:
            stats = {
                "index_name": self.index_name,
                "vectors_count": self.index.ntotal if self.index else 0,
                "created": self.metadata.get("created", 0),
                "updated": self.metadata.get("updated", 0),
                "urls_count": len(self.metadata.get("urls", [])),
                "index_size_bytes": os.path.getsize(self.index_path) if self.index_path.exists() else 0
            }
            
            return stats
        
        except Exception as e:
            logger.error(f"Error getting vector store stats: {str(e)}", exc_info=True)
            return {
                "index_name": self.index_name,
                "error": str(e)
            }
    
    def clear(self) -> bool:
        """
        Clear the vector store.
        
        Returns:
            Boolean indicating success
        """
        try:
            # Reset index and metadata
            self.index = None
            self.metadata = {
                "urls": [],
                "contents": [],
                "created": time.time(),
                "updated": time.time(),
                "count": 0
            }
            self.embeddings = []
            
            # Remove files if they exist
            if self.index_path.exists():
                self.index_path.unlink()
            
            if self.metadata_path.exists():
                self.metadata_path.unlink()
            
            if self.embeddings_path.exists():
                self.embeddings_path.unlink()
            
            logger.info(f"Cleared vector store {self.index_name}")
            return True
        
        except Exception as e:
            logger.error(f"Error clearing vector store: {str(e)}", exc_info=True)
            return False

async def store_vectors(embeddings: List[List[float]], 
                      urls: List[str], 
                      contents: List[str],
                      index_name: str = "default") -> Dict[str, Any]:
    """
    Store vector embeddings in the vector store.
    
    Args:
        embeddings: List of embedding vectors
        urls: List of URLs corresponding to the embeddings
        contents: List of text contents corresponding to the embeddings
        index_name: Name of the vector index
        
    Returns:
        Dictionary with storage results
    """
    try:
        # Create vector store
        vector_store = VectorStore(index_name)
        
        # Add embeddings
        success = vector_store.add_embeddings(embeddings, urls, contents)
        
        if success:
            stats = vector_store.get_stats()
            return {
                "success": True,
                "vectors_added": len(embeddings),
                "total_vectors": stats["vectors_count"],
                "index_name": index_name
            }
        else:
            return {
                "success": False,
                "error": "Failed to add embeddings to vector store"
            }
    
    except Exception as e:
        logger.error(f"Error storing vectors: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }

async def search_vectors(query_embedding: List[float], 
                       top_k: int = 5,
                       index_name: str = "default") -> List[Dict[str, Any]]:
    """
    Search for similar vectors in the vector store.
    
    Args:
        query_embedding: Query vector
        top_k: Number of results to return
        index_name: Name of the vector index
        
    Returns:
        List of dictionaries with search results
    """
    try:
        # Create vector store
        vector_store = VectorStore(index_name)
        
        # Search
        results = vector_store.search(query_embedding, top_k)
        
        return results
    
    except Exception as e:
        logger.error(f"Error searching vectors: {str(e)}", exc_info=True)
        return []

async def get_vector_store_info(index_name: str = "default") -> Dict[str, Any]:
    """
    Get information about the vector store.
    
    Args:
        index_name: Name of the vector index
        
    Returns:
        Dictionary with vector store information
    """
    try:
        # Create vector store
        vector_store = VectorStore(index_name)
        
        # Get stats
        stats = vector_store.get_stats()
        
        return stats
    
    except Exception as e:
        logger.error(f"Error getting vector store info: {str(e)}", exc_info=True)
        return {
            "index_name": index_name,
            "error": str(e)
        }