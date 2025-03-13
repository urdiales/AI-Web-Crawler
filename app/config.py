"""
Configuration Module

Contains application settings and configuration options.
"""

import os
from pathlib import Path
from pydantic import BaseSettings
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

class Settings(BaseSettings):
    """
    Application settings class using Pydantic.
    Environment variables override these defaults.
    """
    
    # Application info
    APP_NAME: str = "AI Web Crawler"
    APP_VERSION: str = "1.0.0"
    
    # Debug mode
    DEBUG: bool = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")
    
    # File paths
    BASE_DIRECTORY: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIRECTORY: str = os.path.join(BASE_DIRECTORY, "data")
    CRAWLED_DATA_DIRECTORY: str = os.path.join(DATA_DIRECTORY, "crawled")
    KNOWLEDGE_BASE_DIRECTORY: str = os.path.join(DATA_DIRECTORY, "knowledge_base")
    EXPORTS_DIRECTORY: str = os.path.join(DATA_DIRECTORY, "exports")
    IMAGES_DIRECTORY: str = os.path.join(DATA_DIRECTORY, "images")
    STATIC_DIRECTORY: str = os.path.join(BASE_DIRECTORY, "static")
    
    # Crawler settings
    USER_AGENT: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    MAX_CRAWL_DEPTH: int = 3
    MAX_PAGES_PER_CRAWL: int = 100
    DEFAULT_WORD_COUNT_THRESHOLD: int = 10
    
    # SharePoint/AD integration
    SHAREPOINT_USERNAME: str = os.getenv("SHAREPOINT_USERNAME", "")
    SHAREPOINT_PASSWORD: str = os.getenv("SHAREPOINT_PASSWORD", "")
    
    # Storage settings
    STORE_IMAGES: bool = True
    MAX_IMAGE_SIZE_MB: int = 10
    
    # Vector database settings
    USE_EMBEDDINGS: bool = os.getenv("USE_EMBEDDINGS", "True").lower() in ("true", "1", "t")
    USE_LOCAL_EMBEDDINGS: bool = os.getenv("USE_LOCAL_EMBEDDINGS", "True").lower() in ("true", "1", "t")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    
    # Ollama settings
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    DEFAULT_OLLAMA_MODEL: str = os.getenv("DEFAULT_OLLAMA_MODEL", "llama3")
    
    # OpenAI (for embeddings if selected)
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    
    # Performance settings
    MAX_CONTEXT_LENGTH: int = 10000  # Max number of characters to send to LLM
    
    def __init__(self, **data):
        super().__init__(**data)
        
        # Ensure directories exist
        for path in [
            self.DATA_DIRECTORY,
            self.CRAWLED_DATA_DIRECTORY,
            self.KNOWLEDGE_BASE_DIRECTORY,
            self.EXPORTS_DIRECTORY,
            self.IMAGES_DIRECTORY
        ]:
            os.makedirs(path, exist_ok=True)

# Create a global instance
SETTINGS = Settings()