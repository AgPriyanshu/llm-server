from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


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