#!/usr/bin/env python3
"""
Create the necessary directory structure for the AI Web Crawler project.
"""

import os
from pathlib import Path

# Define the folder structure
FOLDERS = [
    "app",
    "data",
    "data/crawled",
    "data/knowledge_base",
    "data/exports",
    "data/images",
    "logs",
    "static",
    "static/css",
    "static/img",
]

def create_folder_structure():
    """Create the folder structure if it doesn't exist."""
    print("Creating folder structure...")
    
    # Get the project root directory
    root_dir = Path(__file__).parent.absolute()
    
    # Create each folder
    for folder in FOLDERS:
        folder_path = root_dir / folder
        folder_path.mkdir(parents=True, exist_ok=True)
        print(f"Created: {folder_path}")
    
    print("Folder structure created successfully!")

if __name__ == "__main__":
    create_folder_structure()