"""
Dashboard UI Component

This module handles the main dashboard layout and common UI elements.
"""

import os
import streamlit as st
from pathlib import Path
import base64
import logging

logger = logging.getLogger(__name__)

def setup_dashboard(app_title, app_version):
    """
    Set up the main dashboard layout and styling.
    
    Args:
        app_title (str): Title of the application
        app_version (str): Version of the application
    """
    try:
        # Set page config
        st.set_page_config(
            page_title=app_title,
            page_icon="🕸️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Apply custom CSS
        apply_custom_css()
        
        # Header section with logo
        col1, col2 = st.columns([1, 4])
        
        with col1:
            display_logo()
            
        with col2:
            st.title(app_title)
            st.caption(f"Version {app_version} - Powered by Crawl4AI")
        
        st.divider()
        
        logger.info("Dashboard setup complete")
        
    except Exception as e:
        logger.error(f"Error setting up dashboard: {str(e)}", exc_info=True)
        st.error(f"Error setting up dashboard: {str(e)}")

def apply_custom_css():
    """Apply custom CSS styling to the Streamlit application."""
    custom_css = """
    <style>
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        h1, h2, h3 {
            margin-top: 0;
        }
        .crawl-results {
            margin-top: 2rem;
        }
        .stButton>button {
            width: 100%;
        }
        .company-logo {
            max-width: 150px;
            margin-bottom: 1rem;
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

def display_logo():
    """
    Display company logo if available, otherwise show default icon.
    Logo file should be at 'assets/logo.png'.
    """
    logo_path = Path("assets/logo.png")
    default_icon = "🕸️"
    
    if logo_path.exists():
        try:
            with open(logo_path, "rb") as f:
                logo_bytes = f.read()
            encoded = base64.b64encode(logo_bytes).decode()
            st.markdown(
                f'<img src="data:image/png;base64,{encoded}" class="company-logo">',
                unsafe_allow_html=True
            )
            logger.debug("Custom logo displayed")
        except Exception as e:
            logger.warning(f"Could not load logo: {str(e)}")
            st.markdown(f"<h1>{default_icon}</h1>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h1>{default_icon}</h1>", unsafe_allow_html=True)
        logger.debug("Using default logo icon")

def display_footer():
    """Display application footer with GitHub link."""
    st.sidebar.divider()
    st.sidebar.caption("Knowledge Crawler")
    st.sidebar.caption("© 2024")
    st.sidebar.caption("[GitHub Repository](https://github.com/urdiales/AI-Web-Crawler)")