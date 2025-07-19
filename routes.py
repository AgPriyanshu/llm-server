from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from datetime import datetime

from models import ChatCompletionRequest
from llm_server_core import OllamaLLMServer

# Create router
router = APIRouter()

# Initialize LLM server (will be set by main.py)
llm_server: OllamaLLMServer = None


def set_llm_server(server: OllamaLLMServer):
    """Set the LLM server instance"""
    global llm_server
    llm_server = server


@router.get("/")
async def root():
    """Root endpoint with server information"""
    return {
        "message": "LLM Server with Ollama and MCP",
        "version": "1.0.0",
        "available_models": llm_server.available_models,
        "available_tools": list(llm_server.mcp_server.tools.keys()),
    }


@router.get("/models")
async def list_models():
    """List available Ollama models"""
    return {
        "object": "list",
        "data": [
            {
                "id": model,
                "object": "model",
                "created": int(datetime.now().timestamp()),
                "owned_by": "ollama",
            }
            for model in llm_server.available_models
        ],
    }


@router.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint"""
    if request.stream:
        return StreamingResponse(
            llm_server.stream_chat_completion(request),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )
    else:
        return await llm_server.chat_completion(request)


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "ollama_connected": len(llm_server.available_models) > 0,
    } 