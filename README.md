# AI Web Crawler

A comprehensive knowledge base builder that crawls documentation websites using deep crawling with BFS strategy to systematically explore and extract content. The application filters content to focus on meaningful text blocks while avoiding navigation elements and social media links, then transforms it into clean Markdown format optimized for RAG systems.

## Features

- **Deep Web Crawling**: Systematically explore websites with configurable depth using BFS strategy
- **SharePoint/Active Directory Support**: Crawl SharePoint sites and lists in enterprise environments
- **Content Filtering**: Focus on meaningful content blocks while excluding navigation, headers, etc.
- **Image Support**: Option to include and download images during crawling
- **Knowledge Base Management**: Store and organize crawled content with vector search capabilities
- **Ollama Integration**: Query your knowledge base using Ollama models
- **Modern UI**: Professional Streamlit interface with company logo support
- **Export Options**: Download results in JSON or Markdown formats
- **Docker Support**: Easy deployment with Docker and Docker Compose
- **Robust Error Handling**: Handle network failures, API limits, and more

## Quick Start

### Option 1: Docker Compose (Recommended)

The easiest way to get started is with Docker Compose, which will set up both the AI Web Crawler and Ollama in one command:

1. Clone the repository:

   ```bash
   git clone https://github.com/urdiales/AI-Web-Crawler.git
   cd AI-Web-Crawler
   ```

2. Start the application:

   ```bash
   docker-compose up -d
   ```

3. Access the web interface at:
   ```
   http://localhost:8501
   ```

### Option 2: Manual Setup

If you prefer to run the application without Docker:

1. Clone the repository:

   ```bash
   git clone https://github.com/urdiales/AI-Web-Crawler.git
   cd AI-Web-Crawler
   ```

2. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Install Playwright browsers:

   ```bash
   python -m playwright install --with-deps chromium
   ```

5. Run the application:

   ```bash
   streamlit run app/main.py
   ```

6. Access the web interface at:
   ```
   http://localhost:8501
   ```

## Using the Application

### Crawling Websites

1. Navigate to the **Crawler** tab
2. Enter the URL you want to crawl
3. (Optional) Add keywords to focus content extraction
4. Configure crawl settings:
   - Crawl depth
   - Include images
   - Stay within domain
   - Maximum pages
5. Click "Start Crawling"
6. When crawling completes, you can download the results as JSON or Markdown

### Managing Knowledge Base

1. Navigate to the **Knowledge Base** tab
2. View existing knowledge bases
3. Select a knowledge base to:
   - View its content
   - Export to JSON or Markdown
   - Delete if no longer needed

### Chatting with Your Knowledge

1. Navigate to the **Chat** tab
2. Select a knowledge base to query
3. Choose an Ollama model
4. Ask questions about your crawled content

## Deployment Options

### Docker Compose

The included `docker-compose.yml` provides a complete setup with:

- AI Web Crawler application
- Ollama LLM service
- Persistent volume storage
- GPU support (if available)

### Dockge

To deploy with [Dockge](https://github.com/louislam/dockge):

1. Upload the `docker-compose.yml` file to your Dockge instance
2. Click "Deploy Stack"
3. Access the application at the published port

### Portainer

To deploy with [Portainer](https://www.portainer.io/):

1. Go to your Portainer dashboard
2. Navigate to Stacks → Add stack
3. Upload the `docker-compose.yml` file
4. Deploy the stack
5. Access the application at the published port

## Configuration

The application can be configured using environment variables in the `.env` file or through the Docker Compose file:

| Variable             | Description                         | Default                |
| -------------------- | ----------------------------------- | ---------------------- |
| DEBUG                | Enable debug logging                | False                  |
| USE_EMBEDDINGS       | Enable vector embeddings for search | True                   |
| USE_LOCAL_EMBEDDINGS | Use local embedding models          | True                   |
| EMBEDDING_MODEL      | Model name for embeddings           | all-MiniLM-L6-v2       |
| OLLAMA_HOST          | URL of Ollama API                   | http://localhost:11434 |
| DEFAULT_OLLAMA_MODEL | Default Ollama model                | llama3                 |
| SHAREPOINT_USERNAME  | Username for SharePoint             |                        |
| SHAREPOINT_PASSWORD  | Password for SharePoint             |                        |

## Project Structure

```
AI-Web-Crawler/
├── app/
│   ├── __init__.py
│   ├── main.py                   # Main Streamlit application
│   ├── crawler.py                # Web crawling functionality
│   ├── processor.py              # Content processing and filtering
│   ├── storage.py                # Knowledge base management
│   ├── ollama_integration.py     # Integration with Ollama
│   ├── utils.py                  # Utility functions
│   └── config.py                 # Configuration settings
├── data/                         # Data storage (mapped volume)
├── static/                       # Static assets
├── Dockerfile                    # Docker configuration
├── docker-compose.yml            # Docker Compose configuration
├── requirements.txt              # Python dependencies
└── README.md                     # Documentation
```

## Troubleshooting

### Common Issues

1. **Browser Issues**: If you encounter browser-related errors, try:

   ```bash
   python -m playwright install --with-deps chromium
   ```

2. **Connection to Ollama**: If the application can't connect to Ollama, check:

   - Ollama is running and accessible
   - The OLLAMA_HOST environment variable is set correctly
   - Your network allows connections to the Ollama port

3. **Memory Usage**: For large crawls, increase the container memory limit in the Docker Compose file:
   ```yaml
   deploy:
     resources:
       limits:
         memory: 4G
   ```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
