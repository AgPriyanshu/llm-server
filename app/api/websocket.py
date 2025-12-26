"""WebSocket endpoint for real-time chat."""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from langchain.messages import HumanMessage, SystemMessage

from app.core import logger
from app.services.agent import agent

router = APIRouter()


SYSTEM_PROMPT = [
    "You are a helpful assistant and give precise answers. ",
    "You have access to tools to help you answer questions. ",
    "Mention them in the answer when you use the tools. ",
]


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time chat."""
    await websocket.accept()
    logger.info("WebSocket connection established")

    try:
        while True:
            data = await websocket.receive_text()
            logger.info(f"Received message: {data[:50]}...")

            try:
                messages = [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=data),
                ]

                response = agent.invoke({"messages": messages})
                result_messages = response["messages"]
                await websocket.send_text(result_messages[-1].text)
            except Exception as exc:
                logger.error(f"Error processing WebSocket message: {exc}")
                await websocket.send_text(
                    "Sorry, an error occurred while processing your request."
                )
    except WebSocketDisconnect:
        logger.info("WebSocket connection closed by client")
    except Exception as exc:
        logger.error(f"WebSocket error: {exc}")

