"""
AI Web Crawler

A comprehensive web crawling and knowledge base tool for RAG systems.
"""

__version__ = "1.0.0"

from .crawler import WebCrawler
from .processor import ContentProcessor
from .storage import KnowledgeBase
from .ollama_integration import OllamaAgent
from .config import SETTINGS