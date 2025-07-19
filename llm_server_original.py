import asyncio
import json
import logging
import os
from typing import Dict, List, Optional, Any, AsyncGenerator
from datetime import datetime
import uuid
import re

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import httpx
import ollama
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Pydantic models for API
class ChatMessage(BaseModel):
    role: str = Field(
        ..., description="Role of the message sender (user, assistant, system)"
    )
    content: str = Field(..., description="Content of the message")


class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="Model to use for completion")
    messages: List[ChatMessage] = Field(
        ..., description="List of messages in the conversation"
    )
    stream: bool = Field(default=False, description="Whether to stream the response")
    temperature: Optional[float] = Field(
        default=0.7, description="Temperature for sampling"
    )
    max_tokens: Optional[int] = Field(
        default=None, description="Maximum tokens to generate"
    )
    tools: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="MCP tools available"
    )


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Optional[Dict[str, int]] = None


class MCPTool(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]


class MCPToolCall(BaseModel):
    id: str
    type: str = "function"
    function: Dict[str, Any]


class MCPServer:
    """MCP Server implementation that calls Django backend for tools"""

    def __init__(self):
        self.django_base_url = "http://localhost:80"
        self.tools: Dict[str, MCPTool] = {}
        self._register_default_tools()

    def _register_default_tools(self):
        """Register default MCP tools - schemas only, execution handled by Django"""

        # Calculator tool
        self.register_tool(
            name="calculate",
            description="Perform mathematical calculations",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate",
                    }
                },
                "required": ["expression"],
            },
        )

        # GEOSPATIAL ANALYSIS TOOLS

        # List available layers
        self.register_tool(
            name="list_layers",
            description="List available geospatial layers with their names and IDs",
            parameters={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of layers to return (default: 10, max: 50)",
                        "default": 10,
                    },
                },
            },
        )

        # Find layer by name
        self.register_tool(
            name="find_layer_by_name",
            description="Find a layer by its name and get basic information",
            parameters={
                "type": "object",
                "properties": {
                    "layer_name": {
                        "type": "string",
                        "description": "Name of the layer to find (case-insensitive partial match)",
                    }
                },
                "required": ["layer_name"],
            },
        )

        # Get layer information
        self.register_tool(
            name="get_layer_info",
            description="Get detailed information about a specific layer",
            parameters={
                "type": "object",
                "properties": {
                    "layer_id": {
                        "type": "string",
                        "description": "UUID of the layer to analyze",
                    },
                    "layer_name": {
                        "type": "string",
                        "description": "Name of the layer to analyze (alternative to layer_id)",
                    },
                },
            },
        )

        # HEAP VOLUME ANALYSIS TOOLS

        # List heap clusters
        self.register_tool(
            name="list_heap_clusters",
            description="List heap clusters with volume summaries, optionally filtered by site or iteration",
            parameters={
                "type": "object",
                "properties": {
                    "site_name": {
                        "type": "string",
                        "description": "Filter by site name (case-insensitive partial match)",
                    },
                    "iteration_name": {
                        "type": "string", 
                        "description": "Filter by iteration name (case-insensitive partial match)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of heap clusters to return (default: 10, max: 50)",
                        "default": 10,
                    },
                },
            },
        )

        # Get heap cluster details
        self.register_tool(
            name="get_heap_cluster_details",
            description="Get detailed information about a specific heap cluster including all volume data",
            parameters={
                "type": "object",
                "properties": {
                    "cluster_id": {
                        "type": "string",
                        "description": "UUID of the heap cluster to analyze",
                    },
                    "cluster_name": {
                        "type": "string",
                        "description": "Name of the heap cluster (alternative to cluster_id)",
                    },
                },
            },
        )

        # List heaps in cluster
        self.register_tool(
            name="list_heaps_in_cluster", 
            description="List individual heaps within a specific heap cluster with their volume data",
            parameters={
                "type": "object",
                "properties": {
                    "cluster_id": {
                        "type": "string",
                        "description": "UUID of the heap cluster",
                    },
                    "cluster_name": {
                        "type": "string",
                        "description": "Name of the heap cluster (alternative to cluster_id)",
                    },
                },
            },
        )

        # Get volume summary by site
        self.register_tool(
            name="get_volume_summary_by_site",
            description="Get aggregated volume summary for all heap clusters in a site",
            parameters={
                "type": "object",
                "properties": {
                    "site_name": {
                        "type": "string",
                        "description": "Name of the site to analyze",
                    },
                    "site_id": {
                        "type": "string", 
                        "description": "UUID of the site (alternative to site_name)",
                    },
                },
            },
        )

        # Get heap clusters from multiple iterations
        self.register_tool(
            name="get_heap_clusters_from_multiple_iterations",
            description="Get heap clusters data from multiple iterations at once",
            parameters={
                "type": "object",
                "properties": {
                    "iteration_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of iteration names to fetch heap clusters from",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of heap clusters to return per iteration (default: 10, max: 20)",
                        "default": 10,
                    },
                },
                "required": ["iteration_names"],
            },
        )

    def register_tool(self, name: str, description: str, parameters: Dict[str, Any]):
        """Register a new MCP tool"""
        self.tools[name] = MCPTool(
            name=name, description=description, parameters=parameters
        )
        logger.info(f"Registered MCP tool: {name}")

    async def execute_tool(
        self, tool_name: str, arguments: Dict[str, Any], user_query: str = ""
    ) -> Dict[str, Any]:
        """Execute an MCP tool by calling Django backend"""
        if tool_name not in self.tools:
            return {"success": False, "error": f"Unknown tool: {tool_name}"}

        try:
            async with httpx.AsyncClient(timeout=300.0) as client:  # 5 minutes timeout for database operations
                logger.info(f"Calling Django MCP tool: {tool_name} with arguments: {arguments}")
                
                # Call Django MCP endpoint
                response = await client.post(
                    f"{self.django_base_url}/ai-chat/api/mcp/execute_tool/",
                    json={"tool_name": tool_name, "arguments": arguments, "user_query": user_query},
                )
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"Django MCP tool {tool_name} executed successfully")
                    
                    # Check if the result contains HTML content or chart data from Django
                    if result.get("content_type") == "text/html":
                        logger.info(f"Tool {tool_name} returned HTML content for charts")
                        # Return the HTML content directly as the tool result
                        return {
                            "success": True,
                            "result": result.get("result", ""),
                            "content_type": "text/html"
                        }
                    elif result.get("content_type") == "application/json" and result.get("chart_type"):
                        logger.info(f"Tool {tool_name} returned chart data: {result.get('chart_type')}")
                        # Return the chart data directly as the tool result
                        return {
                            "success": True,
                            "result": result.get("result", ""),
                            "content_type": "application/json",
                            "chart_type": result.get("chart_type")
                        }
                    else:
                        # Return the regular tool result
                        return result
                else:
                    error_msg = f"Django MCP tool {tool_name} failed with status {response.status_code}"
                    logger.error(error_msg)
                    return {"success": False, "error": error_msg}
                    
        except Exception as e:
            error_msg = f"Error executing tool {tool_name}: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}



    def get_tools_schema(self) -> List[Dict[str, Any]]:
        """Get OpenAI-compatible tools schema"""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self.tools.values()
        ]


class OllamaLLMServer:
    """Main LLM Server class with Ollama integration"""

    def __init__(self):
        # Use environment variable for Ollama host, default to localhost
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.ollama_client = ollama.AsyncClient(host=ollama_host)
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
        tools_context = ""
        if request.tools:
            tools_schema = self.mcp_server.get_tools_schema()
            tools_context = f"""
You have access to the following tools. When the user asks for information that requires these tools, you MUST use them by calling the function in this format: function_name(parameter1="value1", parameter2="value2")

Available tools:
"""
            for tool in tools_schema:
                func = tool["function"]
                tools_context += f"- {func['name']}: {func['description']}\n"
                if func.get("parameters", {}).get("properties"):
                    tools_context += f"  Parameters: {', '.join(func['parameters']['properties'].keys())}\n"

            tools_context += """
When a user asks about layers, geospatial data, heap volumes, or analysis, you MUST use the appropriate tools. For example:
- "What layers are available?" -> use list_layers()
- "Find layer X" -> use find_layer_by_name(layer_name="X")
- "Information about layer X" -> use get_layer_info(layer_name="X")
- "Calculate 2+2" -> use calculate(expression="2+2")
- "Show heap clusters" -> use list_heap_clusters()
- "Heap clusters in site X" -> use list_heap_clusters(site_name="X")
- "Heap clusters in iteration Y" -> use list_heap_clusters(iteration_name="Y")
- "Show heap data for iteration Z" -> use list_heap_clusters(iteration_name="Z")
- "Get heap clusters from iterations A, B, C" -> use list_heap_clusters(iteration_name="A") then list_heap_clusters(iteration_name="B") then list_heap_clusters(iteration_name="C")
- "Details of heap cluster X" -> use get_heap_cluster_details(cluster_name="X")
- "Heaps in cluster X" -> use list_heaps_in_cluster(cluster_name="X")
- "Volume summary for site X" -> use get_volume_summary_by_site(site_name="X")
- "Analyze volume data for all heap clusters" -> use list_heap_clusters()
- "Volume breakdown by site" -> use list_heap_clusters()
- "Volume analysis with charts" -> use list_heap_clusters()

IMPORTANT: Always use tools when the user's question requires data that you don't have directly. Never provide placeholder data or instructions on how to implement charts manually. The tools will return actual data with properly formatted charts.

When users mention specific iteration names, always use the iteration_name parameter in list_heap_clusters to filter the results.
When users mention multiple iteration names, use get_heap_clusters_from_multiple_iterations with the iteration_names parameter.
"""
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

            # Process tool calls if present
            assistant_message = response["message"]["content"]
            
            # Clean the response to remove thinking tags
            assistant_message = self._clean_response(assistant_message)

            tool_calls = self._extract_tool_calls(assistant_message)
            
            # Check if user query should trigger list_layers tool (only if tools are available)
            if request.tools and not tool_calls:
                user_query = request.messages[-1].content if request.messages else ""
                if self._should_use_list_layers(user_query):
                    tool_calls = [
                        {
                            "id": str(uuid.uuid4()),
                            "type": "function",
                            "function": {"name": "list_layers", "arguments": {}},
                        }
                    ]
                else:
                    # Check for heap tools
                    heap_tool = self._should_use_heap_tools(user_query)
                    if heap_tool:
                        # Try to extract iteration or site names from the query
                        arguments = self._extract_heap_tool_arguments(user_query)
                        
                        # If multiple iteration names are found, use the multiple iterations tool
                        if "iteration_names" in arguments and len(arguments["iteration_names"]) > 1:
                            heap_tool = "get_heap_clusters_from_multiple_iterations"
                        
                        tool_calls = [
                            {
                                "id": str(uuid.uuid4()),
                                "type": "function",
                                "function": {"name": heap_tool, "arguments": arguments},
                            }
                        ]

            if tool_calls:
                # Execute tool calls
                tool_results = []
                user_query = request.messages[-1].content if request.messages else ""
                has_html_content = False
                has_chart_data = False
                html_content = ""
                chart_data = ""
                chart_type = ""
                
                for tool_call in tool_calls:
                    result = await self.mcp_server.execute_tool(
                        tool_call["function"]["name"],
                        tool_call["function"]["arguments"],
                        user_query,
                    )
                    tool_results.append(result)
                    
                    # Check if any result contains HTML content or chart data
                    # Check the Django API response format: {"data": {"success": True, "result": "<html>", "content_type": "text/html", ...}}
                    data = result.get("data", {})
                    if data.get("success") and data.get("content_type") == "text/html" and data.get("result"):
                        has_html_content = True
                        # Get HTML content from Django API response
                        html_content = data.get("result", "")
                    elif data.get("success") and data.get("content_type") == "application/json" and data.get("chart_type") and data.get("result"):
                        has_chart_data = True
                        # Get chart data from Django API response
                        chart_data = data.get("result", "")
                        chart_type = data.get("chart_type", "")

                # If we have HTML content, return it directly instead of processing through LLM
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
                
                # If we have chart data, return it directly instead of processing through LLM  
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

                # Add tool results to conversation and get final response
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
                assistant_message = self._clean_response(final_response["message"]["content"])

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
            tools_schema = self.mcp_server.get_tools_schema()
            tools_context = f"""

You have access to the following tools. When the user asks for information that requires these tools, you MUST use them by calling the function in this format: function_name(parameter1="value1", parameter2="value2")

Available tools:
"""
            for tool in tools_schema:
                func = tool["function"]
                tools_context += f"- {func['name']}: {func['description']}\n"
                if func.get("parameters", {}).get("properties"):
                    tools_context += f"  Parameters: {', '.join(func['parameters']['properties'].keys())}\n"

            tools_context += """
When a user asks about layers, geospatial data, heap volumes, or analysis, you MUST use the appropriate tools. For example:
- "What layers are available?" -> use list_layers()
- "Find layer X" -> use find_layer_by_name(layer_name="X")
- "Information about layer X" -> use get_layer_info(layer_name="X")
- "Calculate 2+2" -> use calculate(expression="2+2")
- "Show heap clusters" -> use list_heap_clusters()
- "Heap clusters in site X" -> use list_heap_clusters(site_name="X")
- "Heap clusters in iteration Y" -> use list_heap_clusters(iteration_name="Y")
- "Show heap data for iteration Z" -> use list_heap_clusters(iteration_name="Z")
- "Get heap clusters from iterations A, B, C" -> use list_heap_clusters(iteration_name="A") then list_heap_clusters(iteration_name="B") then list_heap_clusters(iteration_name="C")
- "Details of heap cluster X" -> use get_heap_cluster_details(cluster_name="X")
- "Heaps in cluster X" -> use list_heaps_in_cluster(cluster_name="X")
- "Volume summary for site X" -> use get_volume_summary_by_site(site_name="X")
- "Analyze volume data for all heap clusters" -> use list_heap_clusters()
- "Volume breakdown by site" -> use list_heap_clusters()
- "Volume analysis with charts" -> use list_heap_clusters()

IMPORTANT: Always use tools when the user's question requires data that you don't have directly. Never provide placeholder data or instructions on how to implement charts manually. The tools will return actual data with properly formatted charts.

When users mention specific iteration names, always use the iteration_name parameter in list_heap_clusters to filter the results.
"""
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
                
                # Clean the content to remove thinking tags
                content = self._clean_response(content) if content else content

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

    def _clean_response(self, content: str) -> str:
        """Clean the response by removing thinking tags and other unwanted content"""
        # Remove thinking tags and their contents
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        content = re.sub(r'<thinking>.*?</thinking>', '', content, flags=re.DOTALL)
        
        # Remove any remaining XML-like tags that might be thinking related
        content = re.sub(r'</?(?:think|thinking|reason|reasoning|internal)>', '', content, flags=re.IGNORECASE)
        
        # Clean up extra whitespace
        content = re.sub(r'\n\s*\n', '\n\n', content)
        content = content.strip()
        
        return content

    def _extract_tool_calls(self, content: str) -> List[Dict[str, Any]]:
        """Extract tool calls from assistant message"""
        tool_calls = []

        # Look for function calls in various formats
        import re

        # Pattern 1: function_name(arguments)
        pattern1 = r"(\w+)\(([^)]*)\)"
        matches1 = re.findall(pattern1, content)

        # Pattern 2: explicit tool usage mentions
        pattern2 = r"(?:use|call|execute)\s+(\w+)(?:\(([^)]*)\))?"
        matches2 = re.findall(pattern2, content, re.IGNORECASE)

        # Pattern 3: "I'll use X tool" or "Let me check with X"
        pattern3 = r"(?:I\'ll use|let me (?:use|call|check with))\s+(\w+)"
        matches3 = re.findall(pattern3, content, re.IGNORECASE)

        all_matches = []

        # Process pattern 1 matches
        for function_name, args_str in matches1:
            if function_name in self.mcp_server.tools:
                all_matches.append((function_name, args_str))

        # Process pattern 2 matches
        for function_name, args_str in matches2:
            if function_name in self.mcp_server.tools:
                all_matches.append((function_name, args_str or ""))

        # Process pattern 3 matches (no arguments)
        for function_name in matches3:
            if function_name in self.mcp_server.tools:
                all_matches.append((function_name, ""))

        for function_name, args_str in all_matches:
            try:
                # Parse arguments (key=value format)
                args = {}
                if args_str.strip():
                    # Handle both key=value and positional arguments
                    if "=" in args_str:
                        for arg in args_str.split(","):
                            if "=" in arg:
                                key, value = arg.split("=", 1)
                                args[key.strip()] = value.strip().strip("\"'")
                    else:
                        # For simple cases like layer names
                        if function_name in [
                            "find_layer_by_name",
                        ]:
                            args["layer_name"] = args_str.strip().strip("\"'")
                        elif function_name in [
                            "get_heap_cluster_details",
                            "list_heaps_in_cluster",
                        ]:
                            args["cluster_name"] = args_str.strip().strip("\"'")
                        elif function_name in [
                            "get_volume_summary_by_site",
                        ]:
                            args["site_name"] = args_str.strip().strip("\"'")
                        elif function_name in [
                            "list_heap_clusters",
                        ]:
                            # Could be site_name or iteration_name, try to guess from context
                            arg_value = args_str.strip().strip("\"'")
                            if "site" in function_name.lower() or "site" in arg_value.lower():
                                args["site_name"] = arg_value
                            elif "iteration" in function_name.lower() or "iteration" in arg_value.lower():
                                args["iteration_name"] = arg_value
                            else:
                                args["site_name"] = arg_value  # Default to site_name

                tool_calls.append(
                    {
                        "id": str(uuid.uuid4()),
                        "type": "function",
                        "function": {"name": function_name, "arguments": args},
                    }
                )
            except Exception as e:
                logger.error(f"Error parsing tool call: {str(e)}")

        return tool_calls
    
    def _should_use_list_layers(self, user_query: str) -> bool:
        """Check if the user query should trigger list_layers tool"""
        user_query_lower = user_query.lower()
        
        # Look for exact phrases that indicate listing layers
        layer_list_phrases = [
            "list layers",
            "show layers", 
            "what layers",
            "available layers",
            "layers available",
            "which layers",
            "see layers",
            "display layers"
        ]
        
        return any(phrase in user_query_lower for phrase in layer_list_phrases)

    def _should_use_heap_tools(self, user_query: str) -> str:
        """Check if the user query should trigger heap analysis tools and return the appropriate tool"""
        user_query_lower = user_query.lower()
        
        # Look for phrases that indicate heap cluster listing
        heap_list_phrases = [
            "list heap clusters",
            "show heap clusters", 
            "what heap clusters",
            "available heap clusters",
            "heap clusters available",
            "which heap clusters",
            "see heap clusters",
            "display heap clusters",
            "show heaps",
            "list heaps",
            "all heap clusters",
            "heap cluster data",
            "analyze heap clusters",
            "heap cluster analysis"
        ]
        
        # Look for phrases that indicate iteration-specific queries
        iteration_phrases = [
            "iteration",
            "iterations",
            "from iteration",
            "in iteration",
            "for iteration",
            "iteration data",
            "heap data for iteration",
            "clusters in iteration",
            "clusters from iteration",
            "show iteration",
            "get iteration",
            "analyze iteration"
        ]
        
        # Look for phrases that indicate volume summary by site
        volume_summary_phrases = [
            "volume summary",
            "total volume",
            "site volume",
            "volume by site",
            "aggregated volume",
            "volume analysis",
            "volume data",
            "analyze volume",
            "breakdown by site",
            "volume breakdown",
            "site breakdown",
            "volume per site",
            "volume for each site",
            "volume distribution"
        ]
        
        # Look for phrases that indicate heap cluster details
        cluster_detail_phrases = [
            "heap cluster details",
            "cluster information",
            "details of heap cluster",
            "information about cluster",
            "specific heap cluster",
            "individual heap cluster"
        ]
        
        # Priority order: More specific queries first
        
        # Check for iteration-specific queries (highest priority)
        if any(iter_phrase in user_query_lower for iter_phrase in iteration_phrases) and \
           any(heap_phrase in user_query_lower for heap_phrase in ["heap", "cluster", "volume"]):
            return "list_heap_clusters"
        
        # Check for volume analysis with site breakdown (most specific)
        elif any(vol_phrase in user_query_lower for vol_phrase in volume_summary_phrases) and \
             any(heap_phrase in user_query_lower for heap_phrase in ["heap", "cluster"]):
            return "list_heap_clusters"  # Start with listing all heap clusters to get comprehensive data
        
        # Check for heap cluster listing
        elif any(phrase in user_query_lower for phrase in heap_list_phrases):
            return "list_heap_clusters"
        
        # Check for volume summary by site
        elif any(phrase in user_query_lower for phrase in volume_summary_phrases):
            return "get_volume_summary_by_site"
        
        # Check for heap cluster details
        elif any(phrase in user_query_lower for phrase in cluster_detail_phrases):
            return "get_heap_cluster_details"
        
        return None

    def _extract_heap_tool_arguments(self, user_query: str) -> Dict[str, Any]:
        """Extract iteration names, site names, and other arguments from user query"""
        arguments = {}
        user_query_lower = user_query.lower()
        
        # Look for multiple iteration names in various formats
        multi_iteration_patterns = [
            r"iterations?\s+['\"]([^'\"]+)['\"](?:\s*,\s*['\"]([^'\"]+)['\"])*",  # iterations "name1", "name2"
            r"iterations?\s+([A-Za-z0-9_-]+)(?:\s*,\s*([A-Za-z0-9_-]+))*",      # iterations name1, name2
            r"from\s+iterations?\s+['\"]([^'\"]+)['\"](?:\s*,\s*['\"]([^'\"]+)['\"])*",  # from iterations "name1", "name2"
            r"from\s+iterations?\s+([A-Za-z0-9_-]+)(?:\s*,\s*([A-Za-z0-9_-]+))*",      # from iterations name1, name2
        ]
        
        # First try to find multiple iterations
        multiple_iterations = []
        for pattern in multi_iteration_patterns:
            matches = re.findall(pattern, user_query, re.IGNORECASE)
            if matches:
                for match in matches:
                    if isinstance(match, tuple):
                        multiple_iterations.extend([m for m in match if m.strip()])
                    else:
                        multiple_iterations.append(match)
                break
        
        # Also look for comma-separated lists
        comma_separated_patterns = [
            r"iterations?\s+([A-Za-z0-9_-]+(?:\s*,\s*[A-Za-z0-9_-]+)+)",  # iterations name1, name2, name3
            r"from\s+iterations?\s+([A-Za-z0-9_-]+(?:\s*,\s*[A-Za-z0-9_-]+)+)",  # from iterations name1, name2, name3
        ]
        
        for pattern in comma_separated_patterns:
            matches = re.findall(pattern, user_query, re.IGNORECASE)
            if matches:
                for match in matches:
                    names = [name.strip() for name in match.split(',')]
                    multiple_iterations.extend(names)
                break
        
        if multiple_iterations:
            # Clean up the list and remove duplicates
            unique_iterations = list(set([name.strip() for name in multiple_iterations if name.strip()]))
            arguments["iteration_names"] = unique_iterations
        else:
            # Look for single iteration names in various formats
            iteration_patterns = [
                r"iteration\s+['\"]([^'\"]+)['\"]",  # iteration "name"
                r"iteration\s+([A-Za-z0-9_-]+)",     # iteration name
                r"from\s+iteration\s+['\"]([^'\"]+)['\"]",  # from iteration "name"
                r"from\s+iteration\s+([A-Za-z0-9_-]+)",     # from iteration name
                r"in\s+iteration\s+['\"]([^'\"]+)['\"]",    # in iteration "name"
                r"in\s+iteration\s+([A-Za-z0-9_-]+)",       # in iteration name
                r"for\s+iteration\s+['\"]([^'\"]+)['\"]",   # for iteration "name"
                r"for\s+iteration\s+([A-Za-z0-9_-]+)",      # for iteration name
            ]
            
            for pattern in iteration_patterns:
                matches = re.findall(pattern, user_query, re.IGNORECASE)
                if matches:
                    # Take the first match
                    arguments["iteration_name"] = matches[0]
                    break
        
        # Look for site names in various formats (only if no iteration names found)
        if "iteration_name" not in arguments and "iteration_names" not in arguments:
            site_patterns = [
                r"site\s+['\"]([^'\"]+)['\"]",  # site "name"
                r"site\s+([A-Za-z0-9_-]+)",     # site name
                r"from\s+site\s+['\"]([^'\"]+)['\"]",  # from site "name"
                r"from\s+site\s+([A-Za-z0-9_-]+)",     # from site name
                r"in\s+site\s+['\"]([^'\"]+)['\"]",    # in site "name"
                r"in\s+site\s+([A-Za-z0-9_-]+)",       # in site name
                r"for\s+site\s+['\"]([^'\"]+)['\"]",   # for site "name"
                r"for\s+site\s+([A-Za-z0-9_-]+)",      # for site name
            ]
            
            for pattern in site_patterns:
                matches = re.findall(pattern, user_query, re.IGNORECASE)
                if matches:
                    # Take the first match
                    arguments["site_name"] = matches[0]
                    break
        
        return arguments


# Initialize the LLM server
llm_server = OllamaLLMServer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await llm_server.initialize()
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


@app.get("/")
async def root():
    """Root endpoint with server information"""
    return {
        "message": "LLM Server with Ollama and MCP",
        "version": "1.0.0",
        "available_models": llm_server.available_models,
        "available_tools": list(llm_server.mcp_server.tools.keys()),
    }


@app.get("/models")
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


@app.post("/chat/completions")
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


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "ollama_connected": len(llm_server.available_models) > 0,
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8001))

    uvicorn.run(
        "llm_server:app", host="0.0.0.0", port=port, reload=True, log_level="info"
    )
