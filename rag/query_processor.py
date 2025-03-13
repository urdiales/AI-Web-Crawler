"""
Query Processor

This module handles processing of user queries for the RAG system.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple, AsyncGenerator
import time
import asyncio

from .ollama_connector import OllamaConnector
from .retrieval_engine import RetrievalEngine
from database.embeddings import generate_embeddings

logger = logging.getLogger(__name__)

class QueryProcessor:
    """Processes user queries for the RAG system."""
    
    def __init__(self, 
                ollama_url: str = "http://localhost:11434",
                ollama_model: str = "llama3",
                index_name: str = "default"):
        """
        Initialize the query processor.
        
        Args:
            ollama_url: URL for Ollama API
            ollama_model: Model to use
            index_name: Name of the vector index
        """
        self.ollama = OllamaConnector(base_url=ollama_url, model=ollama_model)
        self.retrieval = RetrievalEngine(index_name=index_name)
        
        logger.info(f"Initialized QueryProcessor with model {ollama_model}, index {index_name}")
    
    async def process_query(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Process a user query and generate a response.
        
        Args:
            query: User query
            top_k: Number of context items to retrieve
            
        Returns:
            Dict with response and metadata
        """
        try:
            logger.info(f"Processing query: '{query}'")
            start_time = time.time()
            
            # Retrieve context
            results, context = await self.retrieval.retrieve(query, top_k=top_k)
            
            # Generate response
            response = await self.ollama.generate_rag_response(query, context)
            
            # Calculate timing
            duration = time.time() - start_time
            
            logger.info(f"Query processed in {duration:.2f} seconds")
            
            # Return both response and metadata
            return {
                "query": query,
                "response": response.get("response", ""),
                "context": context,
                "context_sources": [r.get("url", "") for r in results],
                "model": response.get("model", ""),
                "duration_seconds": duration,
                "tokens": response.get("usage", {})
            }
        
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            return {
                "query": query,
                "response": f"Error processing query: {str(e)}",
                "context": [],
                "context_sources": [],
                "error": str(e)
            }
    
    async def process_query_stream(self, query: str, top_k: int = 5) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process a user query and generate a streaming response.
        
        Args:
            query: User query
            top_k: Number of context items to retrieve
            
        Yields:
            Dict with response chunks and metadata
        """
        try:
            logger.info(f"Processing streaming query: '{query}'")
            start_time = time.time()
            
            # Retrieve context (same as non-streaming)
            results, context = await self.retrieval.retrieve(query, top_k=top_k)
            
            # First yield the metadata
            yield {
                "type": "metadata",
                "query": query,
                "context_count": len(context),
                "context_sources": [r.get("url", "") for r in results],
                "start_time": start_time
            }
            
            # Generate streaming response
            async for chunk in self.ollama.generate_rag_response_stream(query, context):
                yield {
                    "type": "chunk",
                    "content": chunk
                }
            
            # Final yield with timing information
            duration = time.time() - start_time
            yield {
                "type": "end",
                "duration_seconds": duration
            }
            
            logger.info(f"Streaming query processed in {duration:.2f} seconds")
        
        except Exception as e:
            logger.error(f"Error processing streaming query: {str(e)}", exc_info=True)
            yield {
                "type": "error",
                "error": str(e)
            }
    
    async def process_direct_query(self, query: str) -> Dict[str, Any]:
        """
        Process a direct query without RAG context.
        
        Args:
            query: User query
            
        Returns:
            Dict with response and metadata
        """
        try:
            logger.info(f"Processing direct query: '{query}'")
            start_time = time.time()
            
            # Generate response without context
            response = await self.ollama.generate(
                prompt=query,
                system_prompt="You are a helpful assistant that provides accurate and detailed information."
            )
            
            # Calculate timing
            duration = time.time() - start_time
            
            logger.info(f"Direct query processed in {duration:.2f} seconds")
            
            return {
                "query": query,
                "response": response.get("response", ""),
                "model": response.get("model", ""),
                "duration_seconds": duration,
                "tokens": response.get("usage", {})
            }
        
        except Exception as e:
            logger.error(f"Error processing direct query: {str(e)}", exc_info=True)
            return {
                "query": query,
                "response": f"Error processing query: {str(e)}",
                "error": str(e)
            }
    
    async def process_json_query(self, query: str, json_path: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Process a query against a specific JSON file.
        
        Args:
            query: User query
            json_path: Path to the JSON file
            top_k: Number of context items to retrieve
            
        Returns:
            Dict with response and metadata
        """
        try:
            logger.info(f"Processing JSON query: '{query}' against {json_path}")
            start_time = time.time()
            
            # Retrieve context from JSON
            results, context = await self.retrieval.retrieve_from_json(query, json_path, top_k=top_k)
            
            # Generate response
            response = await self.ollama.generate_rag_response(query, context)
            
            # Calculate timing
            duration = time.time() - start_time
            
            logger.info(f"JSON query processed in {duration:.2f} seconds")
            
            return {
                "query": query,
                "response": response.get("response", ""),
                "context": context,
                "context_sources": [r.get("title", "") for r in results],
                "model": response.get("model", ""),
                "duration_seconds": duration,
                "tokens": response.get("usage", {})
            }
        
        except Exception as e:
            logger.error(f"Error processing JSON query: {str(e)}", exc_info=True)
            return {
                "query": query,
                "response": f"Error processing query: {str(e)}",
                "context": [],
                "context_sources": [],
                "error": str(e)
            }
    
    async def check_ollama_status(self) -> Dict[str, Any]:
        """
        Check status of Ollama connection and models.
        
        Returns:
            Dict with status information
        """
        try:
            logger.info("Checking Ollama status")
            
            # Check connection
            connection_ok = await self.ollama.check_connection()
            
            # List models
            models = await self.ollama.list_models() if connection_ok else []
            
            return {
                "connected": connection_ok,
                "models": models,
                "model_count": len(models),
                "current_model": self.ollama.model
            }
        
        except Exception as e:
            logger.error(f"Error checking Ollama status: {str(e)}", exc_info=True)
            return {
                "connected": False,
                "error": str(e),
                "models": []
            }