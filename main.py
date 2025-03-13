"""
Knowledge Crawler - Main Application

This is the main entry point for the Knowledge Crawler application.
It initializes the Streamlit interface and orchestrates the different components.
"""

import os
import streamlit as st
from pathlib import Path
import logging
from datetime import datetime

# Import application modules
from ui.dashboard import setup_dashboard
from ui.crawl_ui import render_crawl_interface
from ui.chat_ui import render_chat_interface
from utils.logger import setup_logger
from utils.error_handler import setup_error_handlers

# Set up application constants
APP_TITLE = "Knowledge Crawler"
APP_VERSION = "1.0.0"
DATA_DIR = Path("data")
LOG_DIR = Path("logs")

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)
(DATA_DIR / "markdown").mkdir(exist_ok=True)
(DATA_DIR / "json").mkdir(exist_ok=True)
(DATA_DIR / "images").mkdir(exist_ok=True)
(DATA_DIR / "vector_db").mkdir(exist_ok=True)

# Set up logging
setup_logger(LOG_DIR / f"app_{datetime.now().strftime('%Y-%m-%d')}.log")
logger = logging.getLogger(__name__)

def main():
    """Main application entry point."""
    try:
        logger.info(f"Starting {APP_TITLE} v{APP_VERSION}")
        
        # Set up error handlers
        setup_error_handlers()
        
        # Set up the dashboard and UI
        setup_dashboard(APP_TITLE, APP_VERSION)
        
        # Navigation sidebar
        st.sidebar.title("Navigation")
        page = st.sidebar.radio(
            "Select a page", 
            ["Web Crawler", "Knowledge Chat"],
            key="navigation"
        )
        
        # Display the appropriate page
        if page == "Web Crawler":
            render_crawl_interface()
        elif page == "Knowledge Chat":
            render_chat_interface()
            
        logger.info("Application UI rendered successfully")
            
    except Exception as e:
        logger.error(f"Error in main application: {str(e)}", exc_info=True)
        st.error(f"An error occurred: {str(e)}")
        
if __name__ == "__main__":
    main()