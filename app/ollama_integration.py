"""
Ollama Integration Module

Handles integration with Ollama LLM for querying knowledge bases.
"""

import json
import requests
from typing import Dict, List, Any, Optional, Union
import httpx

from config import SETTINGS
from utils import setup_logger

# Setup logger
logger = setup_logger("ollama")

class OllamaAgent:
    """
    Integration with Ollama for LLM-based question answering.
    """
    
    def __init__(self, ollama_host: str, default_model: str = "llama3"):
        """
        Initialize the Ollama agent.
        
        Args:
            ollama_host: URL of the Ollama instance API
            default_model: Default model to use
        """
        self.ollama_host = ollama_host.rstrip('/')
        self.default_model = default_model
    
    def list_models(self) -> List[str]:
        """
        Get a list of available models from Ollama.
        
        Returns:
            List of model names
        """
        try:
            response = requests.get(f"{self.ollama_host}/api/tags")
            if response.status_code == 200:
                model_data = response.json()
                models = [model["name"] for model in model_data.get("models", [])]
                return models
            else:
                logger.error(f"Failed to get models: {response.status_code}, {response.text}")
                return [self.default_model]  # Return default at minimum
        
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return [self.default_model]  # Return default as fallback
    
    def query(
        self, 
        question: str, 
        context: Dict[str, Any],
        model: Optional[str] = None,
        temperature: float = 0.7
    ) -> str:
        """
        Query Ollama with a question about the context.
        
        Args:
            question: User's question
            context: Knowledge base content
            model: Ollama model to use (defaults to instance default)
            temperature: Temperature for generation (0.0-1.0)
            
        Returns:
            Ollama's response
        """
        if not model:
            model = self.default_model
        
        # Extract relevant information from context
        context_text = self._prepare_context(question, context)
        
        # Create the prompt
        prompt = self._create_prompt(question, context_text)
        
        try:
            # Call Ollama API
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "temperature": temperature,
                    "stream": False
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "Sorry, I couldn't generate a response.")
            else:
                logger.error(f"Ollama API error: {response.status_code}, {response.text}")
                return f"Error: Failed to get response from Ollama (Status: {response.status_code})"
        
        except Exception as e:
            logger.error(f"Error querying Ollama: {e}")
            return f"Error: {str(e)}"
    
    def _prepare_context(self, question: str, context: Dict[str, Any]) -> str:
        """
        Prepare context information for the prompt.
        
        Args:
            question: User's question
            context: Knowledge base content
            
        Returns:
            Formatted context text
        """
        # Get the pages from the knowledge base
        pages = context.get("pages", [])
        
        if not pages:
            return "No relevant information found."
        
        # For each page, extract content, title, and URL
        context_parts = []
        
        for page in pages:
            title = page.get("title", "Untitled")
            url = page.get("url", "")
            content = page.get("content", "")
            
            # Skip if no content
            if not content:
                continue
            
            # Add page info
            context_parts.append(f"--- Document: {title} ---\nSource: {url}\n\n{content}\n")
        
        # Limit context length to avoid token limits
        combined_context = "\n".join(context_parts)
        if len(combined_context) > SETTINGS.MAX_CONTEXT_LENGTH:
            combined_context = combined_context[:SETTINGS.MAX_CONTEXT_LENGTH] + "...(truncated)"
        
        return combined_context
    
    def _create_prompt(self, question: str, context: str) -> str:
        """
        Create a properly formatted prompt for the query.
        
        Args:
            question: User's question
            context: Formatted context text
            
        Returns:
            Formatted prompt
        """
        return f"""
You are a knowledgeable assistant helping to answer questions about specific content.
Use ONLY the following information to answer the question. If you don't know or can't find 
the answer in the provided context, simply say that you don't know based on the available 
information.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:
"""
    
    async def query_stream(
        self, 
        question: str, 
        context: Dict[str, Any],
        model: Optional[str] = None,
        temperature: float = 0.7
    ) -> str:
        """
        Query Ollama with streaming response.
        
        Args:
            question: User's question
            context: Knowledge base content
            model: Ollama model to use (defaults to instance default)
            temperature: Temperature for generation (0.0-1.0)
            
        Returns:
            Complete response string
        """
        if not model:
            model = self.default_model
        
        # Extract relevant information from context
        context_text = self._prepare_context(question, context)
        
        # Create the prompt
        prompt = self._create_prompt(question, context_text)
        
        # Use httpx for async requests
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.post(
                    f"{self.ollama_host}/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "temperature": temperature,
                        "stream": True
                    }
                )
                
                full_response = ""
                
                # Process streaming response
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    
                    try:
                        data = json.loads(line)
                        token = data.get("response", "")
                        full_response += token
                        
                        # If this is the final response
                        if data.get("done", False):
                            break
                    
                    except json.JSONDecodeError:
                        continue
                
                return full_response
            
            except Exception as e:
                logger.error(f"Error streaming from Ollama: {e}")
                return f"Error: {str(e)}"