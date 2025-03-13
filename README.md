# Knowledge Crawler

Knowledge Crawler is a powerful web crawling and RAG (Retrieval-Augmented Generation) application that can systematically explore documentation websites, extract meaningful content, and provide an AI-powered conversational interface to query the knowledge.

![Knowledge Crawler](assets/screenshot.png)

## Features

- **Deep Web Crawling**: Systematically explore websites using BFS strategy with configurable depth
- **Content Filtering**: Focus on meaningful text blocks and remove boilerplate content
- **SharePoint Support**: Crawl SharePoint sites and lists with authentication
- **Image Handling**: Download and process images from websites
- **Structured Output**: Generate clean JSON and Markdown for RAG systems
- **Vector Storage**: Create and query vector embeddings for semantic search
- **Ollama Integration**: Chat with your knowledge base using local Ollama models
- **Professional UI**: Easy-to-use Streamlit interface with company logo support

## Quick Start (Docker)

The easiest way to run Knowledge Crawler is with Docker Compose:

1. Clone the repository:

   ```bash
   git clone https://github.com/urdiales/AI-Web-Crawler.git
   cd AI-Web-Crawler
   ```

2. Create `.env` file with your settings (optional):

   ```bash
   cp .env.example .env
   ```

3. Start with Docker Compose:

   ```bash
   docker-compose -f docker/docker-compose.yml up -d
   ```

4. Access the application at [http://localhost:8501](http://localhost:8501)

The application will automatically download necessary models and start crawling.

## Using Knowledge Crawler

### Crawling a Website

1. Navigate to the "Web Crawler" page
2. Enter the URL you want to crawl
3. Configure advanced options (optional):
   - Focus Keywords
   - Crawl Depth
   - Maximum Pages
   - Image Download
4. Click "Start Crawl"
5. View and download the results

### Chatting with Your Knowledge Base

1. Navigate to the "Knowledge Chat" page
2. Select a knowledge base from the sidebar
3. Enter your questions in the chat input
4. View AI-generated responses based on your crawled content

## Customizing the Application

### Adding a Company Logo

Place your company logo in the `assets/logo.png` file to customize the interface.

### Configuring Ollama

By default, the application connects to Ollama at `http://localhost:11434`. You can change this in:

1. The `.env` file (OLLAMA_BASE_URL)
2. The chat sidebar interface
3. Docker environment variables

### Changing Default Settings

Edit the `.env` file to configure:

- `EMBEDDING_MODEL`: Model for embeddings (local, openai, ollama)
- `OLLAMA_MODEL`: Default Ollama model to use
- `OLLAMA_BASE_URL`: URL of your Ollama instance

## Advanced Deployment Options

### Portainer Deployment

1. In Portainer, go to "Stacks" and click "Add stack"
2. Copy the contents of `docker/docker-compose.yml` into the editor
3. Adjust any environment variables as needed
4. Deploy the stack

### Dockege Deployment

1. In Dockege, click "Add New Template"
2. Create a template with the following settings:
   - Image: ghcr.io/urdiales/knowledge-crawler:latest
   - Port mapping: 8501:8501
   - Environment variables as needed
3. Deploy the template

## Development

### Local Development Setup

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
   python -m playwright install chromium
   ```

5. Run the application:
   ```bash
   cd app
   streamlit run main.py
   ```

### Project Structure

```
knowledge-crawler/
├── app/                      # Application code
│   ├── main.py               # Main application entry point
│   ├── crawler/              # Crawling modules
│   ├── processors/           # Content processing
│   ├── database/             # Vector database
│   ├── rag/                  # RAG components
│   ├── ui/                   # UI components
│   └── utils/                # Utility modules
├── data/                     # Data storage
├── docker/                   # Docker configurations
├── assets/                   # Assets like logos
├── requirements.txt          # Python dependencies
└── README.md                 # Documentation
```

## Troubleshooting

### Common Issues

1. **Ollama Connection Failed**

   - Ensure Ollama is running and accessible
   - Check the URL in the chat sidebar
   - For Docker: ensure the services are on the same network

2. **Crawling Issues**

   - Some sites may block crawlers; try reducing crawl speed
   - For SharePoint: ensure you have access credentials
   - Check logs for detailed error messages

3. **Docker GPU Access**
   - For GPU support, ensure Docker is configured with proper GPU access
   - Check the NVIDIA container toolkit is installed

### Logs

Logs are stored in the `logs` directory:

- `app_YYYY-MM-DD.log`: Main application log
- Check Docker logs with: `docker logs knowledge-crawler`

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [Crawl4AI](https://github.com/unclecode/crawl4ai) for the excellent web crawling functionality
- [Ollama](https://github.com/ollama/ollama) for local LLM support
- [Streamlit](https://streamlit.io/) for the user interface framework
