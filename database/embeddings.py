"""
Embeddings

This module handles the generation of vector embeddings for text content.
"""

import logging
import os
import time
from typing import List, Dict, Any, Optional
import numpy as np
from pathlib import Path
import json
import asyncio

# Try to import various embedding libraries
try:
    import sentence_transformers
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

logger = logging.getLogger(__name__)

# Read configuration from environment
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "local")  # local, openai, ollama
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
LOCAL_MODEL_NAME = os.environ.get("LOCAL_MODEL_NAME", "all-MiniLM-L6-v2")

class EmbeddingGenerator:
    """Handles the generation of embeddings from various providers."""
    
    def __init__(self, model: str = EMBEDDING_MODEL):
        """
        Initialize the embedding generator.
        
        Args:
            model: Embedding model to use ('local', 'openai', 'ollama')
        """
        self.model = model
        self.model_instance = None
        
        # Initialize the appropriate embedding model
        if model == "local" and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                from sentence_transformers import SentenceTransformer
                self.model_instance = SentenceTransformer(LOCAL_MODEL_NAME)
                logger.info(f"Initialized local embedding model: {LOCAL_MODEL_NAME}")
            except Exception as e:
                logger.error(f"Error initializing local embedding model: {str(e)}", exc_info=True)
                self.model = "fallback"
        
        elif model == "openai" and OPENAI_AVAILABLE:
            if not OPENAI_API_KEY:
                logger.warning("OpenAI API key not provided, falling back to local model")
                self.model = "fallback"
            else:
                try:
                    openai.api_key = OPENAI_API_KEY
                    logger.info("Initialized OpenAI embedding model")
                except Exception as e:
                    logger.error(f"Error initializing OpenAI: {str(e)}", exc_info=True)
                    self.model = "fallback"
        
        elif model == "ollama" and OLLAMA_AVAILABLE:
            try:
                import ollama
                # Just verify connectivity to Ollama
                # We'll create actual embeddings on-demand
                ollama.Client(host=OLLAMA_BASE_URL)
                logger.info(f"Initialized Ollama embedding model at {OLLAMA_BASE_URL}")
            except Exception as e:
                logger.error(f"Error initializing Ollama: {str(e)}", exc_info=True)
                self.model = "fallback"
        
        else:
            # Try to fall back to whatever is available
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                try:
                    from sentence_transformers import SentenceTransformer
                    self.model = "local"
                    self.model_instance = SentenceTransformer(LOCAL_MODEL_NAME)
                    logger.info(f"Fallback to local embedding model: {LOCAL_MODEL_NAME}")
                except Exception:
                    logger.error("Could not initialize any embedding model", exc_info=True)
            else:
                logger.error("No embedding models available")
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        try:
            if not texts:
                logger.warning("No texts provided for embedding")
                return []
            
            # Filter out empty texts
            texts = [text for text in texts if text and text.strip()]
            
            if not texts:
                logger.warning("All provided texts were empty")
                return []
            
            # Generate embeddings based on the selected model
            if self.model == "local" and self.model_instance:
                return await self._generate_local_embeddings(texts)
            
            elif self.model == "openai":
                return await self._generate_openai_embeddings(texts)
            
            elif self.model == "ollama":
                return await self._generate_ollama_embeddings(texts)
            
            else:
                logger.error("No embedding model available")
                # Return zeros as fallback
                return [[0.0] * 384 for _ in range(len(texts))]
        
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}", exc_info=True)
            # Return zeros as fallback
            return [[0.0] * 384 for _ in range(len(texts))]
    
    async def _generate_local_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using sentence-transformers."""
        try:
            # Run in thread to avoid blocking event loop
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None, 
                lambda: self.model_instance.encode(texts).tolist()
            )
            
            return embeddings
        
        except Exception as e:
            logger.error(f"Error generating local embeddings: {str(e)}", exc_info=True)
            # Return zeros as fallback
            return [[0.0] * 384 for _ in range(len(texts))]
    
    async def _generate_openai_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI API."""
        try:
            embeddings = []
            
            # Process in batches to respect API limits
            batch_size = 25
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                
                # Create async client and make the request
                from openai import AsyncOpenAI
                client = AsyncOpenAI(api_key=OPENAI_API_KEY)
                response = await client.embeddings.create(
                    model="text-embedding-3-small",
                    input=batch
                )
                
                # Extract embeddings
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                
                # Sleep to respect rate limits
                if i + batch_size < len(texts):
                    await asyncio.sleep(0.5)
            
            return embeddings
        
        except Exception as e:
            logger.error(f"Error generating OpenAI embeddings: {str(e)}", exc_info=True)
            # Return zeros as fallback
            return [[0.0] * 1536 for _ in range(len(texts))]  # OpenAI embeddings are 1536-dim
    
    async def _generate_ollama_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Ollama."""
        try:
            import ollama
            client = ollama.Client(host=OLLAMA_BASE_URL)
            embeddings = []
            
            # Process each text individually
            for text in texts:
                response = client.embeddings(model="llama3", prompt=text)
                embeddings.append(response['embedding'])
            
            return embeddings
        
        except Exception as e:
            logger.error(f"Error generating Ollama embeddings: {str(e)}", exc_info=True)
            # Return zeros as fallback
            return [[0.0] * 4096 for _ in range(len(texts))]  # Most Ollama models use 4096-dim

# Cached embedding generator
_EMBEDDING_GENERATOR = None

async def generate_embeddings(texts: List[str], model: str = EMBEDDING_MODEL) -> List[List[float]]:
    """
    Generate embeddings for a list of texts.
    
    Args:
        texts: List of text strings
        model: Embedding model to use
        
    Returns:
        List of embedding vectors
    """
    global _EMBEDDING_GENERATOR
    
    try:
        # Initialize generator if not already done
        if _EMBEDDING_GENERATOR is None or _EMBEDDING_GENERATOR.model != model:
            _EMBEDDING_GENERATOR = EmbeddingGenerator(model)
        
        # Generate embeddings
        embeddings = await _EMBEDDING_GENERATOR.generate_embeddings(texts)
        
        return embeddings
    
    except Exception as e:
        logger.error(f"Error in generate_embeddings: {str(e)}", exc_info=True)
        # Return zeros as fallback
        return [[0.0] * 384 for _ in range(len(texts))]