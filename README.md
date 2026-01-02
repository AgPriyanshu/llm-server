# LLM Server with Ollama and MCP Support

A comprehensive, modular LLM server that integrates Ollama for local language model inference with Model Context Protocol (MCP) support for tool usage, including advanced geospatial and heap volume analysis. Built with a clean, maintainable architecture using FastAPI.

## Features

- **Ollama Integration**: Connect to local Ollama models
- **MCP Support**: Built-in Model Context Protocol for tool integration
- **OpenAI-Compatible API**: Standard chat completions endpoint
- **Streaming Support**: Real-time response streaming
- **Geospatial Analysis Tools**: List, analyze, and visualize geospatial layers
- **Heap Volume Analysis**: Comprehensive heap cluster and volume data analysis
- **Built-in Tools**: Calculator, file operations, web search, and more
- **FastAPI Backend**: High-performance async API server
- **Modular Architecture**: Clean separation of concerns for maintainability

## Architecture Overview

The LLM Server is built with a modular architecture that separates concerns and makes the codebase highly maintainable. Here's how the components work together:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     main.py     │───▶│   routes.py     │───▶│ llm_server_core │
│  FastAPI App    │    │  HTTP Endpoints │    │  Ollama Client  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│    config.py    │    │   models.py     │    │  mcp_server.py  │
│  Configuration  │    │ Pydantic Models │    │   MCP Tools     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│    utils.py     │    │   __init__.py   │    │   Django API    │
│   Utilities     │    │   Package       │    │   Integration   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Core Components

#### 1. **main.py** - Application Entry Point
- FastAPI application initialization
- CORS middleware configuration
- Application lifespan management
- Server startup and routing setup

#### 2. **models.py** - Data Models
- `ChatMessage`: Individual chat message structure
- `ChatCompletionRequest`: OpenAI-compatible request format
- `ChatCompletionResponse`: Standardized response format
- `MCPTool`: Tool definition schema
- `MCPToolCall`: Tool execution structure

#### 3. **config.py** - Configuration Management
- Environment variable handling
- Ollama and Django service URLs
- Timeout and performance settings
- Tool context templates
- Centralized logging configuration

#### 4. **llm_server_core.py** - Core LLM Logic
- `OllamaLLMServer`: Main server class
- Chat completion handling (streaming and non-streaming)
- Tool call detection and execution
- Response processing and cleaning
- Model management and initialization

#### 5. **mcp_server.py** - Tool Management
- `MCPServer`: MCP protocol implementation
- Tool registration and schema management
- Django backend integration for tool execution
- Geospatial and heap volume analysis tools
- HTTP client management for external APIs

#### 6. **routes.py** - HTTP Endpoints
- `/chat/completions`: OpenAI-compatible chat endpoint
- `/models`: Available model listing
- `/health`: Server health checks
- Root endpoint with server information

#### 7. **utils.py** - Utility Functions
- Response cleaning (removes thinking tags)
- Tool call extraction from text
- Auto-trigger logic for common queries
- Argument parsing for complex tool calls

#### 8. **__init__.py** - Package Interface
- Clean public API exports
- Version information
- Import organization

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
   # Method 1: Direct execution (from llm_server directory)
   cd llm_server
   python main.py
   
   # Method 2: Using uvicorn (from llm_server directory)
   cd llm_server
   uvicorn main:app --host 0.0.0.0 --port 8001 --reload
   
   # Method 3: As a package (from parent directory)
   cd ..
   python -m llm_server
   ```

The server will start on `http://localhost:8001` by default

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
- **calculate**: Perform mathematical calculations

### Geospatial Analysis Tools
- **list_layers**: List all available geospatial layers with names and IDs
- **find_layer_by_name**: Find a layer by its name (case-insensitive partial match)
- **get_layer_info**: Get detailed information about a specific layer

### Heap Volume Analysis Tools
- **list_heap_clusters**: List heap clusters with volume summaries, optionally filtered by site or iteration
- **get_heap_cluster_details**: Get detailed information about a specific heap cluster including all volume data
- **list_heaps_in_cluster**: List individual heaps within a specific heap cluster with their volume data
- **get_volume_summary_by_site**: Get aggregated volume summary for all heap clusters in a site
- **get_heap_clusters_from_multiple_iterations**: Get heap clusters data from multiple iterations at once

## Tool Integration & Auto-Detection

The LLM Server intelligently detects when to use tools based on user queries:

### Auto-Triggered Tools
- **Layer queries**: "What layers are available?" automatically triggers `list_layers`
- **Heap analysis**: "Show heap clusters" automatically triggers `list_heap_clusters`  
- **Site filtering**: "Heap clusters in site X" automatically filters by site
- **Iteration analysis**: "Show data for iteration Y" automatically filters by iteration
- **Volume analysis**: "Volume summary for site Z" triggers volume analysis tools

### Smart Argument Extraction
The system automatically extracts parameters from natural language:
- Site names: "heap clusters in site ABC" → `site_name="ABC"`
- Iteration names: "data from iteration XYZ" → `iteration_name="XYZ"`
- Multiple iterations: "compare iterations A, B, C" → `iteration_names=["A", "B", "C"]`

## Usage Examples

### Basic Chat
```python
import requests

response = requests.post("http://localhost:8001/chat/completions", json={
    "model": "llama2",
    "messages": [
        {"role": "user", "content": "What is the capital of France?"}
    ]
})

print(response.json())
```

### Geospatial Analysis
```python
# Auto-triggered tool usage
response = requests.post("http://localhost:8001/chat/completions", json={
    "model": "llama2", 
    "messages": [
        {"role": "user", "content": "What layers are available?"}
    ],
    "tools": [{}]  # Enable tools
})

# Specific layer analysis
response = requests.post("http://localhost:8001/chat/completions", json={
    "model": "llama2",
    "messages": [
        {"role": "user", "content": "Get information about layer ABC"}
    ],
    "tools": [{}]
})
```

### Heap Volume Analysis
```python
# List all heap clusters
response = requests.post("http://localhost:8001/chat/completions", json={
    "model": "llama2",
    "messages": [
        {"role": "user", "content": "Show me all heap clusters"}
    ],
    "tools": [{}]
})

# Filter by site
response = requests.post("http://localhost:8001/chat/completions", json={
    "model": "llama2",
    "messages": [
        {"role": "user", "content": "Show heap clusters in site MainSite"}
    ],
    "tools": [{}]
})

# Multiple iteration analysis
response = requests.post("http://localhost:8001/chat/completions", json={
    "model": "llama2",
    "messages": [
        {"role": "user", "content": "Compare heap data from iterations Jan2024, Feb2024, Mar2024"}
    ],
    "tools": [{}]
})

# Volume summary
response = requests.post("http://localhost:8001/chat/completions", json={
    "model": "llama2",
    "messages": [
        {"role": "user", "content": "Get volume summary for site MainSite"}
    ],
    "tools": [{}]
})
```

### Streaming Response
```python
import requests

response = requests.post("http://localhost:8001/chat/completions", json={
    "model": "llama2",
    "messages": [
        {"role": "user", "content": "Analyze heap volume trends"}
    ],
    "stream": True,
    "tools": [{}]
}, stream=True)

for line in response.iter_lines():
    if line:
        print(line.decode('utf-8'))
```

## Extending the Architecture

### Adding Custom Tools

Add custom MCP tools by extending the `MCPServer` class in `mcp_server.py`:

```python
# In mcp_server.py
def _register_custom_tools(self):
    """Register custom tools"""
    self.register_tool(
        name="my_custom_tool",
        description="Description of what this tool does",
        parameters={
            "type": "object",
            "properties": {
                "param1": {
                    "type": "string",
                    "description": "Parameter description",
                }
            },
            "required": ["param1"],
        },
    )

# Call this in __init__ after _register_default_tools()
```

### Adding New Utility Functions

Add utility functions in `utils.py`:

```python
# In utils.py
def my_custom_parser(text: str) -> Dict[str, Any]:
    """Parse custom format from text"""
    # Implementation here
    return parsed_data
```

### Custom Configuration

Extend configuration in `config.py`:

```python
# In config.py
MY_CUSTOM_API_URL = os.getenv("MY_CUSTOM_API_URL", "http://localhost:9000")
MY_TIMEOUT = float(os.getenv("MY_TIMEOUT", 60.0))
```

### Custom Routes

Add new endpoints in `routes.py`:

```python
# In routes.py
@router.get("/my-custom-endpoint")
async def my_custom_endpoint():
    """Custom endpoint description"""
    return {"message": "Custom response"}
```

## Configuration

### Environment Variables

The server can be configured using environment variables:

```bash
# Ollama Configuration
export OLLAMA_HOST="http://localhost:11434"

# Server Configuration  
export PORT=8001

# Django Backend (for MCP tools)
# DJANGO_BASE_URL is set in config.py, modify as needed
```

### Default Settings

The server runs with these defaults:
- **Host**: 0.0.0.0 (all interfaces)
- **Port**: 8001 (configurable via PORT env var)
- **Reload**: Enabled in development mode
- **CORS**: Enabled for all origins
- **Timeout**: 300 seconds for tool execution

### File Structure

```
llm_server/
├── main.py              # Application entry point
├── models.py            # Pydantic data models
├── config.py            # Configuration management
├── llm_server_core.py   # Core LLM functionality
├── mcp_server.py        # MCP tool management
├── routes.py            # HTTP endpoint handlers
├── utils.py             # Utility functions
├── __init__.py          # Package initialization
├── requirements.txt     # Python dependencies
├── Dockerfile          # Container configuration
├── README.md           # This documentation
└── llm_server_original.py  # Original monolithic code (preserved)
```

### Development vs Production

**Development:** (run from llm_server directory)
```bash
cd llm_server
python main.py  # Auto-reload enabled
```

**Production:** (run from llm_server directory)
```bash
cd llm_server
uvicorn main:app --host 0.0.0.0 --port 8001 --workers 4
```

**Note:** The import structure is optimized for running directly from the `llm_server` directory. If you need to run as a package, use `python -m llm_server` from the parent directory.

## Benefits of the Modular Architecture

1. **Maintainability**: Each component has a clear, single responsibility
2. **Testability**: Individual modules can be unit tested in isolation  
3. **Scalability**: New features can be added without modifying existing code
4. **Reusability**: Components can be imported and used independently
5. **Debugging**: Issues can be isolated to specific modules
6. **Team Development**: Multiple developers can work on different modules simultaneously

## Migration from Original Code

The original monolithic code has been preserved as `llm_server_original.py`. The refactored code maintains full API compatibility while providing a cleaner, more maintainable structure.