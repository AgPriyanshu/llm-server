import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, AsyncGenerator
from datetime import datetime
import uuid
import re
from urllib.parse import quote_plus

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import httpx
import ollama
from contextlib import asynccontextmanager
from bs4 import BeautifulSoup

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
        self.django_base_url = "http://localhost:8000"
        self.tools: Dict[str, MCPTool] = {}
        self._register_default_tools()

    def _register_default_tools(self):
        """Register default MCP tools - schemas only, execution handled by Django"""

        # Web search tool
        self.register_tool(
            name="web_search",
            description="Search the web for information using DuckDuckGo",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 5)",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        )

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
            description="List all available geospatial layers with their names and IDs",
            parameters={"type": "object", "properties": {}},
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
                        "type": "integer",
                        "description": "ID of the layer to analyze",
                    },
                    "layer_name": {
                        "type": "string",
                        "description": "Name of the layer to analyze (alternative to layer_id)",
                    },
                },
            },
        )

        # Analyze population data
        self.register_tool(
            name="analyze_population",
            description="Analyze population data in a layer and highlight features based on population thresholds",
            parameters={
                "type": "object",
                "properties": {
                    "layer_id": {
                        "type": "integer",
                        "description": "ID of the layer containing population data",
                    },
                    "layer_name": {
                        "type": "string",
                        "description": "Name of the layer containing population data (alternative to layer_id)",
                    },
                    "population_field": {
                        "type": "string",
                        "description": "Name of the field containing population data (e.g., 'population', 'pop_est')",
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Population threshold for highlighting (optional)",
                    },
                    "operation": {
                        "type": "string",
                        "enum": ["greater_than", "less_than", "between", "top_n"],
                        "description": "Type of analysis operation",
                        "default": "greater_than",
                    },
                },
                "required": ["population_field"],
            },
        )

        # Get attribute statistics
        self.register_tool(
            name="get_attribute_stats",
            description="Get statistical analysis of a specific attribute in a layer",
            parameters={
                "type": "object",
                "properties": {
                    "layer_id": {"type": "integer", "description": "ID of the layer"},
                    "layer_name": {
                        "type": "string",
                        "description": "Name of the layer (alternative to layer_id)",
                    },
                    "attribute_key": {
                        "type": "string",
                        "description": "Name of the attribute to analyze",
                    },
                },
                "required": ["attribute_key"],
            },
        )

        # Comprehensive layer analysis
        self.register_tool(
            name="analyze_layer_attributes",
            description="Perform comprehensive attribute analysis on a layer including statistics, data types, and insights",
            parameters={
                "type": "object",
                "properties": {
                    "layer_id": {
                        "type": "integer",
                        "description": "ID of the layer to analyze",
                    },
                    "layer_name": {
                        "type": "string",
                        "description": "Name of the layer to analyze (alternative to layer_id)",
                    },
                    "include_statistics": {
                        "type": "boolean",
                        "description": "Include detailed statistics for numeric attributes",
                        "default": True,
                    },
                },
            },
        )

    def register_tool(self, name: str, description: str, parameters: Dict[str, Any]):
        """Register a new MCP tool"""
        self.tools[name] = MCPTool(
            name=name, description=description, parameters=parameters
        )
        logger.info(f"Registered MCP tool: {name}")

    async def execute_tool(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute an MCP tool by calling Django backend"""
        if tool_name not in self.tools:
            return {"success": False, "error": f"Unknown tool: {tool_name}"}

        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                # Call Django MCP endpoint
                response = await client.post(
                    f"{self.django_base_url}/ai-chat/api/mcp/execute_tool/",
                    json={"tool_name": tool_name, "arguments": arguments},
                )

                if response.status_code == 200:
                    return response.json()
                else:
                    return {
                        "success": False,
                        "error": f"Django MCP call failed: {response.status_code}",
                    }

        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {str(e)}")
            return {"success": False, "error": str(e)}

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
        self.ollama_client = ollama.AsyncClient()
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
When a user asks about layers, geospatial data, or analysis, you should use the appropriate tools. For example:
- "What layers are available?" -> use list_layers()
- "Analyze layer X" -> use find_layer_by_name(layer_name="X") then analyze_layer_attributes(layer_name="X")
- "Population statistics" -> use get_attribute_stats() or analyze_population()

Always use tools when the user's question requires data that you don't have directly.
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

            # Check if we need to force tool usage based on user query
            user_query = messages[-1]["content"].lower()
            forced_tools = self._get_forced_tools(user_query)

            tool_calls = self._extract_tool_calls(assistant_message)

            # Add forced tools if no tools were detected but they should be used
            if not tool_calls and forced_tools:
                tool_calls = forced_tools

            if tool_calls:
                # Execute tool calls
                tool_results = []
                for tool_call in tool_calls:
                    result = await self.mcp_server.execute_tool(
                        tool_call["function"]["name"],
                        tool_call["function"]["arguments"],
                    )
                    tool_results.append(result)

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
                assistant_message = final_response["message"]["content"]

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
When a user asks about layers, geospatial data, or analysis, you should use the appropriate tools. For example:
- "What layers are available?" -> use list_layers()
- "Analyze layer X" -> use find_layer_by_name(layer_name="X") then analyze_layer_attributes(layer_name="X")
- "Population statistics" -> use get_attribute_stats() or analyze_population()

Always use tools when the user's question requires data that you don't have directly.
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

        # Also check if the user's question implies tool usage
        content_lower = content.lower()
        if any(
            phrase in content_lower
            for phrase in ["what layers", "available layers", "list layers"]
        ):
            if not any(match[0] == "list_layers" for match in all_matches):
                all_matches.append(("list_layers", ""))

        if any(
            phrase in content_lower for phrase in ["analyze", "analysis", "attributes"]
        ):
            # Look for layer names in quotes or after "layer"
            layer_match = re.search(
                r'(?:layer\s+["\']?(\w+)["\']?|["\']([^"\']+)["\']?\s+layer)',
                content_lower,
            )
            if layer_match and not any(
                match[0].startswith("analyze") for match in all_matches
            ):
                layer_name = layer_match.group(1) or layer_match.group(2)
                all_matches.append(
                    ("analyze_layer_attributes", f'layer_name="{layer_name}"')
                )

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
                            "analyze_layer_attributes",
                        ]:
                            args["layer_name"] = args_str.strip().strip("\"'")

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

    def _get_forced_tools(self, user_query: str) -> List[Dict[str, Any]]:
        """Force tool usage based on user query patterns"""
        forced_tools = []

        # Force list_layers for layer discovery queries
        if any(
            phrase in user_query
            for phrase in [
                "what layers",
                "available layers",
                "list layers",
                "show layers",
                "layers are available",
                "which layers",
                "what maps",
                "available maps",
            ]
        ):
            forced_tools.append(
                {
                    "id": str(uuid.uuid4()),
                    "type": "function",
                    "function": {"name": "list_layers", "arguments": {}},
                }
            )

        # Force layer analysis for analysis queries
        elif any(
            phrase in user_query for phrase in ["analyze", "analysis", "attributes"]
        ):
            # Try to extract layer name
            import re

            layer_patterns = [
                r'(?:layer\s+["\']?(\w+)["\']?)',
                r'(?:["\']([^"\']+)["\']?\s+layer)',
                r'(?:of\s+(?:the\s+)?["\']?([^"\']+?)["\']?(?:\s+layer)?)',
                r"(?:the\s+([a-zA-Z_]+)\s+(?:layer|data))",
            ]

            layer_name = None
            for pattern in layer_patterns:
                match = re.search(pattern, user_query, re.IGNORECASE)
                if match:
                    layer_name = match.group(1)
                    break

            if layer_name:
                forced_tools.append(
                    {
                        "id": str(uuid.uuid4()),
                        "type": "function",
                        "function": {
                            "name": "analyze_layer_attributes",
                            "arguments": {"layer_name": layer_name},
                        },
                    }
                )
            else:
                # If no specific layer, list available layers first
                forced_tools.append(
                    {
                        "id": str(uuid.uuid4()),
                        "type": "function",
                        "function": {"name": "list_layers", "arguments": {}},
                    }
                )

        # Force population analysis for population queries
        elif any(
            phrase in user_query
            for phrase in ["population", "demographic", "people", "inhabitants"]
        ):
            if any(phrase in user_query for phrase in ["statistics", "stats", "data"]):
                # Try to extract layer name
                import re

                layer_match = re.search(
                    r"(?:of|in|for)\s+(?:the\s+)?([a-zA-Z_]+)",
                    user_query,
                    re.IGNORECASE,
                )
                if layer_match:
                    layer_name = layer_match.group(1)
                    forced_tools.append(
                        {
                            "id": str(uuid.uuid4()),
                            "type": "function",
                            "function": {
                                "name": "get_attribute_stats",
                                "arguments": {
                                    "layer_name": layer_name,
                                    "attribute_key": "population",
                                },
                            },
                        }
                    )

        return forced_tools


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

    uvicorn.run(
        "llm_server:app", host="0.0.0.0", port=8001, reload=True, log_level="info"
    )
