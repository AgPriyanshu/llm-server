"""Pydantic schemas for API request/response models."""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""

    message: str = Field(..., min_length=1, description="The user's message")


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""

    ai_response: str = Field(..., description="The AI assistant's response")


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""

    status: str = Field(default="healthy", description="Service health status")
    version: str = Field(default="1.0.0", description="API version")

