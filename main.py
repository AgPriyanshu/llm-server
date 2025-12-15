import asyncio
from email.mime import base

from fastapi import FastAPI, WebSocket
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt
from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langchain.tools import tool
from langchain_ollama import ChatOllama

from vector_db import VectorDB, vector_db

# @tool(response_format="content_and_artifact")
# def retrieve_mininig_knowledge(query: str):
#     """Retrieve information from mining book to help answer the query."""
#     retrieved_docs = vector_db.vector_store.similarity_search(query, k=2)
#     serialized = "\n\n".join(
#         (f"Source: {doc.metadata}\nContent: {doc.page_content}")
#         for doc in retrieved_docs
#     )
#     return serialized, retrieved_docs


@tool
def get_weather(location: str) -> str:
    """Get weather information for a location."""
    return f"Weather in {location}: Sunny, 72°F"


@dynamic_prompt
def prompt_with_context(request: ModelRequest) -> str:
    """Inject context into state messages."""
    last_query = request.state["messages"][-1].text
    retrieved_docs = vector_db.vector_store.similarity_search(last_query)

    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

    system_message = (
        "You are a helpful assistant. Use the following context in your response:"
        f"\n\n{docs_content}"
    )

    return system_message


model = ChatOllama(
    model="qwen3:4b-instruct", base_url="http://host.docker.internal:11434"
)


agent = create_agent(
    model=model,
    tools=[get_weather],
    middleware=[prompt_with_context],
)

app = FastAPI()


@app.on_event("startup")
async def startup_event():
    # run the heavy sync init in a thread so we don't block the event loop
    loop = asyncio.get_running_loop()
    db = VectorDB()

    def sync_init():
        # you can also supply env-based config here
        db._ensure_initialized()  # or call a dedicated db.init_client(...) method
        return True

    try:
        await loop.run_in_executor(None, sync_init)
        print("VectorDB initialized")
    except Exception as exc:
        print("Failed to init VectorDB: %s", exc)
        # Depending on choice, re-raise to stop startup or handle gracefully
        raise


@app.get("/")
def ping():
    messages = [
        SystemMessage(
            content=[
                "You are a helpful assistant and give precise answers. ",
                "You have access to tools to help you answer questions. ",
                "Use them when necessary but don't mention them in your answers. ",
            ]
        ),
        HumanMessage(content="what is the weather in India"),
    ]

    response = agent.invoke({"messages": messages})
    return {"ai_response": response}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    while True:
        data = await websocket.receive_text()
        messages = [
            SystemMessage(
                content=[
                    "You are a helpful assistant and give precise answers. ",
                    "You have access to tools to help you answer questions. ",
                    "Mention them in the answer when you use the tools. ",
                ]
            ),
            HumanMessage(content=data),
        ]

        response = agent.invoke({"messages": messages})
        messages = response["messages"]
        await websocket.send_text(messages[-1].text)
