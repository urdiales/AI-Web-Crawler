"""
Ollama Connector

This module handles communication with Ollama for LLM generation.
"""

import logging
import os
import json
import time
from typing import List, Dict, Any, Optional, Generator, AsyncGenerator
import aiohttp
import asyncio
import requests

logger = logging.getLogger(__name__)

# Read Ollama configuration from environment
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "llama3")

class OllamaConnector:
    """Handles communication with Ollama API."""
    
    def __init__(self, base_url: str = OLLAMA_BASE_URL, model: str = DEFAULT_MODEL):
        """
        Initialize the Ollama connector.
        
        Args:
            base_url: Base URL for Ollama API
            model: Default model to use
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        
        logger.info(f"Initialized Ollama connector with base URL: {base_url}, model: {model}")
    
    async def check_connection(self) -> bool:
        """
        Check connection to Ollama API.
        
        Returns:
            Boolean indicating success
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/version") as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"Connected to Ollama {data.get('version', 'unknown version')}")
                        return True
                    else:
                        logger.warning(f"Failed to connect to Ollama: HTTP {response.status}")
                        return False
        
        except Exception as e:
            logger.error(f"Error connecting to Ollama: {str(e)}", exc_info=True)
            return False
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """
        List available models.
        
        Returns:
            List of model information dictionaries
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/tags") as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('models', [])
                    else:
                        logger.warning(f"Failed to list models: HTTP {response.status}")
                        return []
        
        except Exception as e:
            logger.error(f"Error listing models: {str(e)}", exc_info=True)
            return []
    
    async def generate(self, 
                     prompt: str, 
                     model: Optional[str] = None,
                     system_prompt: Optional[str] = None,
                     temperature: float = 0.7,
                     max_tokens: int = 2048) -> Dict[str, Any]:
        """
        Generate a response from the model.
        
        Args:
            prompt: User prompt
            model: Model to use (overrides default)
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Dictionary with the response
        """
        try:
            model_name = model or self.model
            
            # Prepare the request
            payload = {
                "model": model_name,
                "prompt": prompt,
                "temperature": temperature,
                "num_predict": max_tokens,
            }
            
            # Add system prompt if provided
            if system_prompt:
                payload["system"] = system_prompt
            
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.base_url}/api/generate", json=payload) as response:
                    if response.status == 200:
                        # Parse response
                        data = await response.json()
                        return {
                            "response": data.get("response", ""),
                            "model": model_name,
                            "created": time.time(),
                            "usage": {
                                "prompt_tokens": data.get("prompt_eval_count", 0),
                                "completion_tokens": data.get("eval_count", 0),
                                "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
                            }
                        }
                    else:
                        error_text = await response.text()
                        logger.error(f"Error from Ollama API: {error_text}")
                        return {
                            "error": f"Ollama API error: {response.status}",
                            "details": error_text,
                            "model": model_name
                        }
        
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}", exc_info=True)
            return {
                "error": f"Error: {str(e)}",
                "model": model or self.model
            }
    
    async def generate_stream(self, 
                            prompt: str, 
                            model: Optional[str] = None,
                            system_prompt: Optional[str] = None,
                            temperature: float = 0.7,
                            max_tokens: int = 2048) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response from the model.
        
        Args:
            prompt: User prompt
            model: Model to use (overrides default)
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Yields:
            Generated text chunks
        """
        try:
            model_name = model or self.model
            
            # Prepare the request
            payload = {
                "model": model_name,
                "prompt": prompt,
                "temperature": temperature,
                "num_predict": max_tokens,
                "stream": True
            }
            
            # Add system prompt if provided
            if system_prompt:
                payload["system"] = system_prompt
            
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.base_url}/api/generate", 
                                       json=payload, 
                                       timeout=aiohttp.ClientTimeout(total=300)) as response:
                    if response.status == 200:
                        # Process the streaming response
                        async for line in response.content:
                            if line:
                                try:
                                    data = json.loads(line)
                                    if "response" in data:
                                        yield data["response"]
                                    
                                    # Check for end of stream
                                    if data.get("done", False):
                                        break
                                except json.JSONDecodeError:
                                    logger.warning(f"Failed to decode JSON: {line}")
                    else:
                        error_text = await response.text()
                        logger.error(f"Error from Ollama API: {error_text}")
                        yield f"Error: {response.status}"
        
        except Exception as e:
            logger.error(f"Error generating streaming response: {str(e)}", exc_info=True)
            yield f"Error: {str(e)}"
    
    async def generate_rag_response(self, 
                                  query: str, 
                                  context: List[str],
                                  model: Optional[str] = None,
                                  system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a response with RAG context.
        
        Args:
            query: User query
            context: List of context strings
            model: Model to use (overrides default)
            system_prompt: Optional system prompt
            
        Returns:
            Dictionary with the response
        """
        try:
            # Combine context
            combined_context = "\n\n".join(context)
            
            # Create RAG prompt
            rag_prompt = f"""Context information:
```
{combined_context}
```

Based on the context information provided, answer the following query:
{query}

If the context doesn't contain relevant information to answer the query, just say so. Don't make up information that's not in the context."""
            
            # Default system prompt for RAG if not provided
            if not system_prompt:
                system_prompt = """You are a helpful assistant that answers questions based on the provided context.
Follow these guidelines:
1. Only use information from the provided context to answer the question
2. If the context doesn't contain the answer, say "I don't have enough information to answer that question"
3. Keep answers concise but comprehensive
4. Format your answers appropriately using Markdown for readability
5. For code samples, use appropriate syntax highlighting
6. Cite specific parts of the context when appropriate"""
            
            # Generate response
            response = await self.generate(
                prompt=rag_prompt,
                model=model,
                system_prompt=system_prompt,
                temperature=0.3,  # Lower temperature for factuality
                max_tokens=2048
            )
            
            return response
        
        except Exception as e:
            logger.error(f"Error generating RAG response: {str(e)}", exc_info=True)
            return {
                "error": f"Error: {str(e)}",
                "model": model or self.model
            }
    
    async def generate_rag_response_stream(self, 
                                         query: str, 
                                         context: List[str],
                                         model: Optional[str] = None,
                                         system_prompt: Optional[str] = None) -> AsyncGenerator[str, None]:
        """
        Generate a streaming RAG response.
        
        Args:
            query: User query
            context: List of context strings
            model: Model to use (overrides default)
            system_prompt: Optional system prompt
            
        Yields:
            Generated text chunks
        """
        try:
            # Combine context
            combined_context = "\n\n".join(context)
            
            # Create RAG prompt
            rag_prompt = f"""Context information:
```
{combined_context}
```

Based on the context information provided, answer the following query:
{query}

If the context doesn't contain relevant information to answer the query, just say so. Don't make up information that's not in the context."""
            
            # Default system prompt for RAG if not provided
            if not system_prompt:
                system_prompt = """You are a helpful assistant that answers questions based on the provided context.
Follow these guidelines:
1. Only use information from the provided context to answer the question
2. If the context doesn't contain the answer, say "I don't have enough information to answer that question"
3. Keep answers concise but comprehensive
4. Format your answers appropriately using Markdown for readability
5. For code samples, use appropriate syntax highlighting
6. Cite specific parts of the context when appropriate"""
            
            # Generate streaming response
            async for chunk in self.generate_stream(
                prompt=rag_prompt,
                model=model,
                system_prompt=system_prompt,
                temperature=0.3,  # Lower temperature for factuality
                max_tokens=2048
            ):
                yield chunk
        
        except Exception as e:
            logger.error(f"Error generating streaming RAG response: {str(e)}", exc_info=True)
            yield f"Error: {str(e)}"
