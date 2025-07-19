import asyncio
import json
import uuid
from typing import AsyncGenerator
from datetime import datetime

import ollama
from fastapi import HTTPException

from models import ChatCompletionRequest, ChatCompletionResponse
from mcp_server import MCPServer
from config import OLLAMA_HOST, TOOLS_CONTEXT_TEMPLATE, logger
from utils import (
    clean_response, 
    extract_tool_calls, 
    should_use_list_layers, 
    should_use_heap_tools,
    extract_heap_tool_arguments
)


class OllamaLLMServer:
    """Main LLM Server class with Ollama integration"""

    def __init__(self):
        self.ollama_client = ollama.AsyncClient(host=OLLAMA_HOST)
        self.mcp_server = MCPServer()
        self.available_models = []

    async def initialize(self):
        """Initialize the server and load available models"""
        try:
            models = await self.ollama_client.list()
            self.available_models = [model["name"] for model in models["models"]]
            logger.info(f"Available Ollama models: {self.available_models}")
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {str(e)}")
            self.available_models = []

    async def chat_completion(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """Handle chat completion request"""
        if request.model not in self.available_models:
            raise HTTPException(
                status_code=400, detail=f"Model {request.model} not available"
            )

        # Convert messages to Ollama format
        messages = [
            {"role": msg.role, "content": msg.content} for msg in request.messages
        ]

        # Add tools to the conversation if provided
        if request.tools:
            tools_context = self._build_tools_context()
            messages[-1]["content"] += tools_context

        try:
            response = await self.ollama_client.chat(
                model=request.model,
                messages=messages,
                options={
                    "temperature": request.temperature,
                    "num_predict": request.max_tokens,
                },
            )

            # Process the response
            assistant_message = response["message"]["content"]
            assistant_message = clean_response(assistant_message)

            # Handle tool calls
            tool_calls = extract_tool_calls(assistant_message, self.mcp_server.tools)
            
            # Check if we should auto-trigger tools
            if request.tools and not tool_calls:
                tool_calls = self._check_auto_trigger_tools(request.messages)

            if tool_calls:
                return await self._handle_tool_calls(
                    tool_calls, request, messages, assistant_message
                )

            return ChatCompletionResponse(
                id=str(uuid.uuid4()),
                created=int(datetime.now().timestamp()),
                model=request.model,
                choices=[
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": assistant_message},
                        "finish_reason": "stop",
                    }
                ],
            )

        except Exception as e:
            logger.error(f"Error in chat completion: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    async def stream_chat_completion(
        self, request: ChatCompletionRequest
    ) -> AsyncGenerator[str, None]:
        """Handle streaming chat completion request"""
        if request.model not in self.available_models:
            raise HTTPException(
                status_code=400, detail=f"Model {request.model} not available"
            )

        messages = [
            {"role": msg.role, "content": msg.content} for msg in request.messages
        ]

        # Add tools context if provided
        if request.tools:
            tools_context = self._build_tools_context()
            messages[-1]["content"] += tools_context

        try:
            stream = await self.ollama_client.chat(
                model=request.model,
                messages=messages,
                stream=True,
                options={
                    "temperature": request.temperature,
                    "num_predict": request.max_tokens,
                },
            )

            chunk_id = str(uuid.uuid4())

            async for chunk in stream:
                content = chunk["message"]["content"]
                content = clean_response(content) if content else content

                response_chunk = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": int(datetime.now().timestamp()),
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": content},
                            "finish_reason": None,
                        }
                    ],
                }

                # Only yield if there's actual content after cleaning
                if content:
                    yield f"data: {json.dumps(response_chunk)}\n\n"

            # Send final chunk
            final_chunk = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": int(datetime.now().timestamp()),
                "model": request.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }

            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Error in streaming completion: {str(e)}")
            error_chunk = {"error": {"message": str(e), "type": "server_error"}}
            yield f"data: {json.dumps(error_chunk)}\n\n"

    def _build_tools_context(self) -> str:
        """Build tools context for the conversation"""
        tools_schema = self.mcp_server.get_tools_schema()
        tools_list = ""
        for tool in tools_schema:
            func = tool["function"]
            tools_list += f"- {func['name']}: {func['description']}\n"
            if func.get("parameters", {}).get("properties"):
                tools_list += f"  Parameters: {', '.join(func['parameters']['properties'].keys())}\n"
        
        return TOOLS_CONTEXT_TEMPLATE.format(tools_list=tools_list)

    def _check_auto_trigger_tools(self, messages) -> list:
        """Check if we should automatically trigger tools based on user query"""
        user_query = messages[-1].content if messages else ""
        
        if should_use_list_layers(user_query):
            return [
                {
                    "id": str(uuid.uuid4()),
                    "type": "function",
                    "function": {"name": "list_layers", "arguments": {}},
                }
            ]
        
        heap_tool = should_use_heap_tools(user_query)
        if heap_tool:
            arguments = extract_heap_tool_arguments(user_query)
            
            # If multiple iteration names are found, use the multiple iterations tool
            if "iteration_names" in arguments and len(arguments["iteration_names"]) > 1:
                heap_tool = "get_heap_clusters_from_multiple_iterations"
            
            return [
                {
                    "id": str(uuid.uuid4()),
                    "type": "function",
                    "function": {"name": heap_tool, "arguments": arguments},
                }
            ]
        
        return []

    async def _handle_tool_calls(self, tool_calls, request, messages, assistant_message):
        """Handle execution of tool calls and return appropriate response"""
        tool_results = []
        user_query = request.messages[-1].content if request.messages else ""
        has_html_content = False
        has_chart_data = False
        html_content = ""
        chart_data = ""
        
        for tool_call in tool_calls:
            result = await self.mcp_server.execute_tool(
                tool_call["function"]["name"],
                tool_call["function"]["arguments"],
                user_query,
            )
            tool_results.append(result)
            
            # Check for special content types
            data = result.get("data", {})
            if data.get("success") and data.get("content_type") == "text/html" and data.get("result"):
                has_html_content = True
                html_content = data.get("result", "")
            elif data.get("success") and data.get("content_type") == "application/json" and data.get("chart_type") and data.get("result"):
                has_chart_data = True
                chart_data = data.get("result", "")

        # Return HTML content directly if present
        if has_html_content:
            return ChatCompletionResponse(
                id=str(uuid.uuid4()),
                created=int(datetime.now().timestamp()),
                model=request.model,
                choices=[
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant", 
                            "content": html_content,
                            "content_type": "text/html"
                        },
                        "finish_reason": "stop",
                    }
                ],
            )
        
        # Return chart data directly if present
        if has_chart_data:
            return ChatCompletionResponse(
                id=str(uuid.uuid4()),
                created=int(datetime.now().timestamp()),
                model=request.model,
                choices=[
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant", 
                            "content": chart_data,
                            "content_type": "application/json"
                        },
                        "finish_reason": "stop",
                    }
                ],
            )

        # Get final response from LLM with tool results
        messages.append({"role": "assistant", "content": assistant_message})
        messages.append(
            {
                "role": "user",
                "content": f"Tool execution results: {json.dumps(tool_results, indent=2)}",
            }
        )

        final_response = await self.ollama_client.chat(
            model=request.model,
            messages=messages,
            options={
                "temperature": request.temperature,
                "num_predict": request.max_tokens,
            },
        )
        
        final_content = clean_response(final_response["message"]["content"])

        return ChatCompletionResponse(
            id=str(uuid.uuid4()),
            created=int(datetime.now().timestamp()),
            model=request.model,
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": final_content},
                    "finish_reason": "stop",
                }
            ],
        ) 