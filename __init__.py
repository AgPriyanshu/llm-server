"""
LLM Server Package

A comprehensive LLM server with Ollama integration and Model Context Protocol support.
"""

from llm_server_core import OllamaLLMServer
from mcp_server import MCPServer
from models import (
    ChatMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    MCPTool,
    MCPToolCall,
)
from main import app

__version__ = "1.0.0"
__author__ = "Your Organization"

__all__ = [
    "OllamaLLMServer",
    "MCPServer", 
    "ChatMessage",
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "MCPTool",
    "MCPToolCall",
    "app",
] 