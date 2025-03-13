"""
Chat UI

This module handles the chat interface for interacting with the knowledge base.
"""

import streamlit as st
import os
import logging
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import asyncio

from rag.query_processor import QueryProcessor
from rag.retrieval_engine import RetrievalEngine
from ui.dashboard import display_footer

logger = logging.getLogger(__name__)

def render_chat_interface():
    """Render the chat interface for the knowledge base."""
    try:
        st.header("Knowledge Chat")
        
        # Initialize session state for chat
        initialize_chat_state()
        
        # Sidebar with options
        render_chat_sidebar()
        
        # Main chat area
        render_chat_messages()
        
        # Input area
        render_chat_input()
        
        # Footer
        display_footer()
    
    except Exception as e:
        logger.error(f"Error rendering chat interface: {str(e)}", exc_info=True)
        st.error(f"An error occurred: {str(e)}")

def initialize_chat_state():
    """Initialize session state variables for the chat."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "ollama_model" not in st.session_state:
        st.session_state.ollama_model = "llama3"
    
    if "ollama_url" not in st.session_state:
        st.session_state.ollama_url = "http://localhost:11434"
    
    if "context_size" not in st.session_state:
        st.session_state.context_size = 5
    
    if "selected_crawl" not in st.session_state:
        st.session_state.selected_crawl = None
    
    if "show_context" not in st.session_state:
        st.session_state.show_context = False

def render_chat_sidebar():
    """Render the sidebar with chat options."""
    try:
        st.sidebar.subheader("Chat Settings")
        
        # Ollama URL
        ollama_url = st.sidebar.text_input(
            "Ollama URL",
            value=st.session_state.ollama_url,
            help="URL of the Ollama instance (e.g., http://localhost:11434)",
            key="ollama_url_input"
        )
        
        # Check Ollama connection button
        if st.sidebar.button("Check Connection", key="check_ollama"):
            check_ollama_connection(ollama_url)
        
        # Model selection
        if "available_models" in st.session_state and st.session_state.available_models:
            model_options = [model["name"] for model in st.session_state.available_models]
            selected_model = st.sidebar.selectbox(
                "Ollama Model",
                options=model_options,
                index=model_options.index(st.session_state.ollama_model) if st.session_state.ollama_model in model_options else 0,
                key="model_select"
            )
            st.session_state.ollama_model = selected_model
        else:
            ollama_model = st.sidebar.text_input(
                "Ollama Model",
                value=st.session_state.ollama_model,
                help="Name of the Ollama model to use",
                key="model_input"
            )
            st.session_state.ollama_model = ollama_model
        
        # Context size
        context_size = st.sidebar.slider(
            "Context Size",
            min_value=1,
            max_value=10,
            value=st.session_state.context_size,
            help="Number of context chunks to retrieve",
            key="context_size_slider"
        )
        st.session_state.context_size = context_size
        
        # Show context toggle
        show_context = st.sidebar.checkbox(
            "Show Context",
            value=st.session_state.show_context,
            help="Show the retrieved context along with the response",
            key="show_context_checkbox"
        )
        st.session_state.show_context = show_context
        
        # Knowledge base selection
        st.sidebar.subheader("Knowledge Base")
        
        # Get available crawls
        crawls = asyncio.run(get_available_crawls())
        
        if crawls:
            # Format options for selectbox
            crawl_options = ["None (Direct LLM)"] + [f"{crawl['domain']} - {crawl['date']} ({crawl['page_count']} pages)" for crawl in crawls]
            
            selected_index = 0
            if st.session_state.selected_crawl:
                # Find matching crawl in list
                for i, crawl in enumerate(crawls):
                    if crawl["id"] == st.session_state.selected_crawl["id"]:
                        selected_index = i + 1  # +1 because of "None" option
                        break
            
            selected_option = st.sidebar.selectbox(
                "Select Knowledge Base",
                options=crawl_options,
                index=selected_index,
                key="crawl_select"
            )
            
            # Update selected crawl
            if selected_option == "None (Direct LLM)":
                st.session_state.selected_crawl = None
            else:
                index = crawl_options.index(selected_option) - 1  # -1 because of "None" option
                st.session_state.selected_crawl = crawls[index]
        else:
            st.sidebar.info("No knowledge bases available. Try crawling a website first.")
            st.session_state.selected_crawl = None
        
        # Clear chat button
        if st.sidebar.button("Clear Chat", key="clear_chat"):
            st.session_state.messages = []
            st.rerun()
    
    except Exception as e:
        logger.error(f"Error rendering chat sidebar: {str(e)}", exc_info=True)
        st.sidebar.error(f"Error: {str(e)}")

def render_chat_messages():
    """Render the chat message history."""
    try:
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Show context if enabled and available
                if st.session_state.show_context and message["role"] == "assistant" and "context" in message:
                    with st.expander("Context Used"):
                        for i, ctx in enumerate(message["context"]):
                            st.markdown(f"**Source {i+1}:**")
                            st.markdown(ctx)
    
    except Exception as e:
        logger.error(f"Error rendering chat messages: {str(e)}", exc_info=True)
        st.error(f"Error displaying chat: {str(e)}")

def render_chat_input():
    """Render the chat input area."""
    try:
        # Chat input
        if prompt := st.chat_input("Ask about the crawled content..."):
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                
                # Use spinner for response generation
                with st.spinner("Thinking..."):
                    # Process the query
                    if st.session_state.selected_crawl:
                        # RAG response with knowledge base
                        response = asyncio.run(process_rag_query(
                            prompt, 
                            st.session_state.selected_crawl,
                            st.session_state.ollama_url,
                            st.session_state.ollama_model,
                            st.session_state.context_size
                        ))
                        full_response = response["response"]
                        context = response.get("context", [])
                    else:
                        # Direct LLM response
                        response = asyncio.run(process_direct_query(
                            prompt,
                            st.session_state.ollama_url,
                            st.session_state.ollama_model
                        ))
                        full_response = response["response"]
                        context = []
                
                # Display full response
                message_placeholder.markdown(full_response)
                
                # Store assistant message
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": full_response,
                    "context": context
                })
    
    except Exception as e:
        logger.error(f"Error in chat input processing: {str(e)}", exc_info=True)
        st.error(f"An error occurred: {str(e)}")

async def process_rag_query(query: str, 
                           crawl_info: Dict[str, Any],
                           ollama_url: str,
                           ollama_model: str,
                           context_size: int) -> Dict[str, Any]:
    """
    Process a RAG query against a knowledge base.
    
    Args:
        query: User query
        crawl_info: Information about the selected crawl
        ollama_url: URL of the Ollama instance
        ollama_model: Name of the Ollama model
        context_size: Number of context chunks to retrieve
        
    Returns:
        Dict with response and metadata
    """
    try:
        logger.info(f"Processing RAG query: '{query}'")
        
        # Create query processor
        processor = QueryProcessor(
            ollama_url=ollama_url,
            ollama_model=ollama_model
        )
        
        # Different processing based on crawl type
        if "file_path" in crawl_info and os.path.exists(crawl_info["file_path"]):
            # Using JSON file directly
            response = await processor.process_json_query(
                query=query,
                json_path=crawl_info["file_path"],
                top_k=context_size
            )
        else:
            # Using vector store
            response = await processor.process_query(
                query=query,
                top_k=context_size
            )
        
        return response
    
    except Exception as e:
        logger.error(f"Error processing RAG query: {str(e)}", exc_info=True)
        return {
            "response": f"Error: {str(e)}",
            "context": [],
            "error": str(e)
        }

async def process_direct_query(query: str, 
                              ollama_url: str,
                              ollama_model: str) -> Dict[str, Any]:
    """
    Process a direct query to the LLM without RAG context.
    
    Args:
        query: User query
        ollama_url: URL of the Ollama instance
        ollama_model: Name of the Ollama model
        
    Returns:
        Dict with response and metadata
    """
    try:
        logger.info(f"Processing direct query: '{query}'")
        
        # Create query processor
        processor = QueryProcessor(
            ollama_url=ollama_url,
            ollama_model=ollama_model
        )
        
        # Process direct query
        response = await processor.process_direct_query(query)
        
        return response
    
    except Exception as e:
        logger.error(f"Error processing direct query: {str(e)}", exc_info=True)
        return {
            "response": f"Error: {str(e)}",
            "error": str(e)
        }

def check_ollama_connection(ollama_url: str):
    """
    Check connection to Ollama and retrieve available models.
    
    Args:
        ollama_url: URL of the Ollama instance
    """
    try:
        # Create query processor
        processor = QueryProcessor(ollama_url=ollama_url)
        
        # Check status
        status = asyncio.run(processor.check_ollama_status())
        
        if status["connected"]:
            st.session_state.ollama_url = ollama_url
            st.session_state.available_models = status["models"]
            
            st.sidebar.success(f"Connected to Ollama! Found {len(status['models'])} models.")
            
            # List available models
            if status["models"]:
                model_list = ", ".join([model["name"] for model in status["models"][:5]])
                if len(status["models"]) > 5:
                    model_list += f" and {len(status['models']) - 5} more"
                st.sidebar.info(f"Available models: {model_list}")
        else:
            st.sidebar.error(f"Failed to connect to Ollama at {ollama_url}")
    
    except Exception as e:
        logger.error(f"Error checking Ollama connection: {str(e)}", exc_info=True)
        st.sidebar.error(f"Error connecting to Ollama: {str(e)}")

async def get_available_crawls() -> List[Dict[str, Any]]:
    """
    Get a list of available crawls for the knowledge base.
    
    Returns:
        List of crawl information dictionaries
    """
    try:
        # Create retrieval engine
        retrieval = RetrievalEngine()
        
        # Get recent crawls
        crawls = await retrieval.get_recent_crawls(limit=10)
        
        return crawls
    
    except Exception as e:
        logger.error(f"Error getting available crawls: {str(e)}", exc_info=True)
        return []