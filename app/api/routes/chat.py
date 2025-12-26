"""Chat endpoint for synchronous requests."""

from fastapi import APIRouter, HTTPException
from langchain.messages import HumanMessage, SystemMessage

from app.core import logger
from app.models import ChatRequest, ChatResponse
from app.services.agent import agent

router = APIRouter(prefix="/chat", tags=["Chat"])


SYSTEM_PROMPT = [
    "You are a helpful assistant and give precise answers. ",
    "You have access to tools to help you answer questions. ",
    "Use them when necessary but don't mention them in your answers. ",
]


@router.post("", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Send a message and receive an AI response."""
    try:
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=request.message),
        ]

        response = agent.invoke({"messages": messages})
        ai_message = response["messages"][-1]
        return ChatResponse(ai_response=ai_message.text)
    except Exception as exc:
        logger.error(f"Chat error: {exc}")
        raise HTTPException(status_code=500, detail="Failed to process chat request")

