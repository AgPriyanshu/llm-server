# LLM Server with Ollama and MCP Support

A comprehensive LLM server that integrates Ollama for local language model inference with Model Context Protocol (MCP) support for tool usage, including advanced geospatial analysis.

## Features

- **Ollama Integration**: Connect to local Ollama models
- **MCP Support**: Built-in Model Context Protocol for tool integration
- **OpenAI-Compatible API**: Standard chat completions endpoint
- **Streaming Support**: Real-time response streaming
- **Geospatial Analysis Tools**: List, analyze, and visualize geospatial layers
- **Built-in Tools**: File operations, web search, calculator, and more
- **FastAPI Backend**: High-performance async API server

## Prerequisites

1. **Install Ollama**: Download and install from [ollama.ai](https://ollama.ai)
2. **Pull Models**: Download models you want to use
   ```bash
   ollama pull llama2
   ollama pull mistral
   ollama pull codellama
   ```

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Start the server:
   ```bash
   python llm_server.py
   ```

The server will start on `http://localhost:8000`

## API Endpoints

### Chat Completions
```bash
POST /chat/completions
```

OpenAI-compatible endpoint for chat completions, with tool support.

**Example Request:**
```json
{
  "model": "llama2",
  "messages": [
    {"role": "user", "content": "What layers are available?"}
  ],
  "tools": true,
  "stream": false
}
```

### List Models
```bash
GET /models
```

Returns available Ollama models.

### List Tools
```bash
GET /tools
```

Returns available MCP tools (including geospatial tools).

### Execute Tool
```bash
POST /mcp/execute_tool?tool_name=<tool_name>
```

Execute an MCP tool directly.

### Health Check
```bash
GET /health
```

Server health status.

## Built-in MCP Tools

### General Tools
- **read_file**: Read file contents
- **web_search**: Search the web using DuckDuckGo
- **get_page_content**: Fetch and extract text content from any web page
- **validate_url**: Check if a URL is accessible and get basic information
- **calculate**: Perform mathematical calculations

### Geospatial Tools
- **list_layers**: List all available geospatial layers
- **find_layer_by_name**: Find a layer by its name
- **get_layer_info**: Get detailed information about a specific layer
- **analyze_population**: Analyze population data in a layer
- **filter_features**: Filter features in a layer by attribute conditions
- **get_attribute_stats**: Get statistics for a specific attribute in a layer
- **spatial_analysis**: Perform spatial operations (buffer, intersection, etc.)
- **create_map_visualization**: Create a map visualization with styled features
- **analyze_layer_attributes**: Comprehensive attribute analysis for a layer

## Enhanced Web Search Features

The web search tool now provides real browser-based search capabilities:

- **DuckDuckGo Integration**: Uses both API and HTML scraping for comprehensive results
- **Instant Answers**: Gets direct answers for factual queries
- **Definitions**: Provides definitions for terms and concepts  
- **Related Topics**: Suggests related information and links
- **Fallback Search**: HTML scraping when API doesn't return results
- **Clean Results**: Properly formatted with titles, snippets, and URLs

### Web Search Examples

```python
# Basic web search
{
  "model": "llama2",
  "messages": [
    {"role": "user", "content": "Search for information about quantum computing"}
  ],
  "tools": ["web_search"]
}

# Get page content from a specific URL
{
  "model": "llama2", 
  "messages": [
    {"role": "user", "content": "Get the content from https://example.com/article"}
  ],
  "tools": ["get_page_content"]
}

# Validate a URL before accessing
{
  "model": "llama2",
  "messages": [
    {"role": "user", "content": "Check if https://example.com is accessible"}
  ],
  "tools": ["validate_url"]
}
```

## Usage Examples

### Basic Chat
```python
import requests

response = requests.post("http://localhost:8000/chat/completions", json={
    "model": "llama2",
    "messages": [
        {"role": "user", "content": "What is the capital of France?"}
    ]
})

print(response.json())
```

### Geospatial Analysis Example
```python
response = requests.post("http://localhost:8000/chat/completions", json={
    "model": "llama2",
    "messages": [
        {"role": "user", "content": "Analyze the attributes of the tl 2024 36 place layer"}
    ],
    "tools": true
})
print(response.json())
```

### Direct Tool Execution
```python
response = requests.post(
    "http://localhost:8000/mcp/execute_tool?tool_name=list_layers",
    json={}
)
print(response.json())
```

## Extending with Custom Tools

You can add custom MCP tools by extending the `MCPServer` class in `llm_server.py`. See the code for examples.

## Configuration

The server runs on:
- **Host**: 0.0.0.0 (all interfaces)
- **Port**: 8000
- **Reload**: Enabled in development mode

Modify the `uvicorn.run()` call in `