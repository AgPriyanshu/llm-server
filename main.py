#!/usr/bin/env python3
"""
LLM Server with Ollama and MCP Integration

This is the main entry point for the LLM server application.
"""

import uvicorn
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from llm_server_core import OllamaLLMServer
from routes import router, set_llm_server
from config import PORT


# Initialize the LLM server
llm_server = OllamaLLMServer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    await llm_server.initialize()
    set_llm_server(llm_server)
    yield
    # Shutdown
    pass


# Create FastAPI app
app = FastAPI(
    title="LLM Server with Ollama and MCP",
    description="A comprehensive LLM server with Ollama integration and Model Context Protocol support",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router)


if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=PORT, 
        reload=True, 
        log_level="info"
    ) 