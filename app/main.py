import os
import streamlit as st
from pathlib import Path
import pandas as pd
import json
import base64
from PIL import Image
import io
import time
from datetime import datetime

# Import application modules
from crawler import WebCrawler
from processor import ContentProcessor
from storage import KnowledgeBase
from ollama_integration import OllamaAgent
from config import SETTINGS
from utils import setup_logger, create_directory_if_not_exists

# Setup logging
logger = setup_logger()

# Initialize directories
for directory in [
    SETTINGS.DATA_DIRECTORY,
    SETTINGS.CRAWLED_DATA_DIRECTORY,
    SETTINGS.KNOWLEDGE_BASE_DIRECTORY,
    SETTINGS.IMAGES_DIRECTORY
]:
    create_directory_if_not_exists(directory)

# Set page configuration
st.set_page_config(
    page_title="AI Web Crawler",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
def load_css():
    with open(Path(SETTINGS.STATIC_DIRECTORY) / "css" / "style.css") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Try to load CSS
try:
    load_css()
except Exception as e:
    logger.warning(f"Could not load custom CSS: {e}")

# App state management
if 'crawl_results' not in st.session_state:
    st.session_state.crawl_results = None
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "Crawler"
if 'knowledge_base' not in st.session_state:
    st.session_state.knowledge_base = KnowledgeBase(SETTINGS.KNOWLEDGE_BASE_DIRECTORY)
if 'ollama_agent' not in st.session_state:
    st.session_state.ollama_agent = OllamaAgent(SETTINGS.OLLAMA_HOST, SETTINGS.DEFAULT_OLLAMA_MODEL)
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Sidebar for logo and navigation
with st.sidebar:
    # Logo upload or default
    st.title("AI Web Crawler")
    
    # Logo upload section
    uploaded_logo = st.file_uploader("Upload Company Logo", type=["png", "jpg", "jpeg"])
    if uploaded_logo is not None:
        # Display uploaded logo
        logo_image = Image.open(uploaded_logo)
        st.image(logo_image, width=200)
        
        # Save logo for future use
        logo_path = Path(SETTINGS.IMAGES_DIRECTORY) / "company_logo.png"
        logo_image.save(logo_path)
    else:
        # Check if we have a saved logo
        logo_path = Path(SETTINGS.IMAGES_DIRECTORY) / "company_logo.png"
        if logo_path.exists():
            st.image(str(logo_path), width=200)
        else:
            # Default logo/text
            st.markdown("### 🕸️ Knowledge Extractor")
    
    # Navigation
    st.sidebar.markdown("## Navigation")
    tabs = ["Crawler", "Knowledge Base", "Chat"]
    selected_tab = st.sidebar.radio("Select a tab:", tabs, index=tabs.index(st.session_state.current_tab))
    st.session_state.current_tab = selected_tab
    
    # Settings section
    with st.sidebar.expander("Settings"):
        ollama_host = st.text_input("Ollama Host URL", value=SETTINGS.OLLAMA_HOST)
        ollama_model = st.text_input("Default Ollama Model", value=SETTINGS.DEFAULT_OLLAMA_MODEL)
        if st.button("Update Settings"):
            SETTINGS.OLLAMA_HOST = ollama_host
            SETTINGS.DEFAULT_OLLAMA_MODEL = ollama_model
            # Reinitialize the Ollama agent with new settings
            st.session_state.ollama_agent = OllamaAgent(SETTINGS.OLLAMA_HOST, SETTINGS.DEFAULT_OLLAMA_MODEL)
            st.success("Settings updated!")

# Helper function for file download
def get_download_link(file_path, link_text):
    with open(file_path, 'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{os.path.basename(file_path)}">{link_text}</a>'
    return href

# Helper function to generate timestamp string
def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

# Crawler Tab
if st.session_state.current_tab == "Crawler":
    st.title("Web Crawler")
    st.markdown("Extract knowledge from websites for your RAG system.")
    
    # Crawling form
    with st.form("crawl_form"):
        url = st.text_input("URL to crawl:", placeholder="https://example.com")
        
        col1, col2 = st.columns(2)
        with col1:
            keywords = st.text_area("Keywords (optional, one per line):", 
                                    placeholder="Enter keywords to focus on specific content")
            max_depth = st.slider("Crawl Depth", min_value=1, max_value=5, value=2, 
                                   help="How many levels deep to crawl from the starting URL")
        
        with col2:
            include_images = st.checkbox("Include Images", value=True)
            stay_within_domain = st.checkbox("Stay Within Domain", value=True)
            max_pages = st.number_input("Max Pages", min_value=1, max_value=1000, value=50,
                                        help="Maximum number of pages to crawl")
        
        submitted = st.form_submit_button("Start Crawling")
    
    if submitted:
        if not url:
            st.error("Please enter a URL to crawl.")
        else:
            # Process keywords if provided
            keyword_list = None
            if keywords:
                keyword_list = [k.strip() for k in keywords.split('\n') if k.strip()]
            
            # Initialize crawler with form options
            crawler = WebCrawler(
                max_depth=max_depth,
                include_images=include_images,
                stay_within_domain=stay_within_domain,
                max_pages=max_pages,
                keywords=keyword_list
            )
            
            # Show progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Start crawling with progress updates
                status_text.text("Initializing crawler...")
                
                # Start crawling with progress callback
                def update_progress(current, total, message):
                    if total > 0:
                        progress = min(current / total, 1.0)
                        progress_bar.progress(progress)
                        status_text.text(f"{message} ({current}/{total})")
                    else:
                        status_text.text(message)
                
                # Execute the crawl
                results = crawler.crawl(url, progress_callback=update_progress)
                
                # Process the crawled content
                status_text.text("Processing content...")
                processor = ContentProcessor()
                processed_results = processor.process(results)
                
                # Store in session state
                st.session_state.crawl_results = processed_results
                
                # Save results
                timestamp = get_timestamp()
                
                # Save as JSON
                json_path = Path(SETTINGS.CRAWLED_DATA_DIRECTORY) / f"crawl_{timestamp}.json"
                with open(json_path, 'w') as f:
                    json.dump(processed_results, f, indent=2)
                
                # Save as Markdown
                md_path = Path(SETTINGS.CRAWLED_DATA_DIRECTORY) / f"crawl_{timestamp}.md"
                markdown_content = processor.to_markdown(processed_results)
                with open(md_path, 'w') as f:
                    f.write(markdown_content)
                
                # Add to knowledge base
                kb = st.session_state.knowledge_base
                kb_id = kb.add_content(processed_results, f"crawl_{timestamp}")
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                # Show success message with download links
                st.success(f"Successfully crawled {len(processed_results['pages'])} pages!")
                
                # Display download links
                st.markdown(get_download_link(json_path, "Download JSON"), unsafe_allow_html=True)
                st.markdown(get_download_link(md_path, "Download Markdown"), unsafe_allow_html=True)
                
                # Preview the results
                with st.expander("Preview Results"):
                    st.write(processed_results)
                    
                    # Create DataFrame for better visualization
                    if processed_results and 'pages' in processed_results:
                        pages_df = pd.DataFrame([
                            {
                                "URL": page['url'],
                                "Title": page['title'],
                                "Word Count": len(page['content'].split()),
                                "Depth": page.get('depth', 0)
                            }
                            for page in processed_results['pages']
                        ])
                        
                        st.subheader("Crawled Pages")
                        st.dataframe(pages_df)
                
            except Exception as e:
                # Handle errors
                st.error(f"Error during crawling: {str(e)}")
                logger.error(f"Crawling error: {e}", exc_info=True)
                progress_bar.empty()
                status_text.empty()

# Knowledge Base Tab
elif st.session_state.current_tab == "Knowledge Base":
    st.title("Knowledge Base")
    st.markdown("Manage your knowledge base of crawled content.")
    
    kb = st.session_state.knowledge_base
    
    # Get existing knowledge bases
    kb_list = kb.list_knowledge_bases()
    
    if not kb_list:
        st.info("No knowledge bases found. Crawl some websites to create a knowledge base.")
    else:
        # Display knowledge bases
        st.subheader("Available Knowledge Bases")
        
        # Create a DataFrame
        kb_df = pd.DataFrame([
            {
                "ID": kb_id,
                "Name": info['name'],
                "Created": info['created'],
                "Pages": info['page_count']
            }
            for kb_id, info in kb_list.items()
        ])
        
        st.dataframe(kb_df)
        
        # Select knowledge base for viewing/management
        selected_kb = st.selectbox(
            "Select a knowledge base to manage:",
            options=list(kb_list.keys()),
            format_func=lambda x: f"{kb_list[x]['name']} ({kb_list[x]['page_count']} pages)"
        )
        
        if selected_kb:
            # Show knowledge base details
            kb_details = kb.get_knowledge_base(selected_kb)
            
            st.subheader(f"Knowledge Base: {kb_details['name']}")
            
            # Actions for this knowledge base
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Export as JSON"):
                    export_path = kb.export_knowledge_base(selected_kb, 'json')
                    st.markdown(get_download_link(export_path, "Download JSON"), unsafe_allow_html=True)
                
                if st.button("Export as Markdown"):
                    export_path = kb.export_knowledge_base(selected_kb, 'markdown')
                    st.markdown(get_download_link(export_path, "Download Markdown"), unsafe_allow_html=True)
            
            with col2:
                if st.button("Delete this Knowledge Base", key=f"delete_{selected_kb}"):
                    if kb.delete_knowledge_base(selected_kb):
                        st.success("Knowledge base deleted successfully.")
                        st.experimental_rerun()
                    else:
                        st.error("Failed to delete knowledge base.")
            
            # Show pages in this knowledge base
            st.subheader("Pages in Knowledge Base")
            
            if 'pages' in kb_details and kb_details['pages']:
                pages_df = pd.DataFrame([
                    {
                        "URL": page['url'],
                        "Title": page['title'],
                        "Word Count": len(page['content'].split())
                    }
                    for page in kb_details['pages']
                ])
                
                st.dataframe(pages_df)
                
                # View page content
                selected_page_idx = st.selectbox(
                    "Select a page to view:",
                    options=range(len(kb_details['pages'])),
                    format_func=lambda x: kb_details['pages'][x]['title']
                )
                
                if st.button("View Page Content"):
                    selected_page = kb_details['pages'][selected_page_idx]
                    st.subheader(selected_page['title'])
                    st.markdown(f"**URL:** {selected_page['url']}")
                    
                    if 'images' in selected_page and selected_page['images'] and include_images:
                        st.subheader("Images")
                        cols = st.columns(3)
                        for i, img in enumerate(selected_page['images']):
                            with cols[i % 3]:
                                st.image(img['src'], caption=img.get('alt', ''), width=200)
                                st.caption(img['src'])
                    
                    st.subheader("Content")
                    st.markdown(selected_page['content'])
            else:
                st.info("No pages found in this knowledge base.")

# Chat Tab
elif st.session_state.current_tab == "Chat":
    st.title("Chat with Your Knowledge")
    st.markdown("Ask questions about your crawled content.")
    
    # Initialize components if needed
    kb = st.session_state.knowledge_base
    ollama_agent = st.session_state.ollama_agent
    
    # Get available knowledge bases for selection
    kb_list = kb.list_knowledge_bases()
    
    if not kb_list:
        st.info("No knowledge bases found. Please crawl websites first.")
    else:
        # Select knowledge base and model
        col1, col2 = st.columns(2)
        
        with col1:
            selected_kb = st.selectbox(
                "Select a knowledge base:",
                options=list(kb_list.keys()),
                format_func=lambda x: f"{kb_list[x]['name']} ({kb_list[x]['page_count']} pages)"
            )
        
        with col2:
            available_models = ollama_agent.list_models()
            selected_model = st.selectbox(
                "Select an Ollama model:",
                options=available_models,
                index=available_models.index(SETTINGS.DEFAULT_OLLAMA_MODEL) if SETTINGS.DEFAULT_OLLAMA_MODEL in available_models else 0
            )
        
        # Create columns for chat UI
        chat_container = st.container()
        
        # Input for questions
        with st.form("chat_form"):
            user_question = st.text_input("Ask a question about the crawled content:", placeholder="What is...?")
            col1, col2 = st.columns([3, 1])
            
            with col1:
                temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1,
                                      help="Higher values make responses more creative, lower values more deterministic")
            
            with col2:
                submit_btn = st.form_submit_button("Ask")
        
        # Handle question submission
        if submit_btn and user_question:
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            
            # Get selected knowledge base
            kb_data = kb.get_knowledge_base(selected_kb)
            
            # Get answer from Ollama
            with st.spinner("Thinking..."):
                try:
                    answer = ollama_agent.query(
                        question=user_question,
                        context=kb_data,
                        model=selected_model,
                        temperature=temperature
                    )
                    
                    # Add assistant response to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Error getting response: {str(e)}")
                    logger.error(f"Ollama error: {e}", exc_info=True)
        
        # Display chat history
        with chat_container:
            if st.session_state.chat_history:
                for message in st.session_state.chat_history:
                    if message["role"] == "user":
                        st.markdown(f"**You:** {message['content']}")
                    else:
                        st.markdown(f"**AI:** {message['content']}")
                        
                # Clear chat button
                if st.button("Clear Chat"):
                    st.session_state.chat_history = []
                    st.experimental_rerun()
            else:
                st.info("Ask a question about your knowledge base to start chatting.")

# Footer
st.markdown("---")
st.markdown(
    "AI Web Crawler - A powerful tool for building knowledge bases from websites. "
    "Created with Streamlit and Crawl4AI."
)