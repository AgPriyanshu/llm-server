from fastapi import FastAPI
from langchain.agents import create_agent
from langchain.messages import HumanMessage, SystemMessage
from langchain.tools import tool
from langchain_ollama import ChatOllama


@tool
def get_weather(location: str) -> str:
    """Get weather information for a location."""
    return f"Weather in {location}: Sunny, 72°F"


model = ChatOllama(model="qwen3:4b-instruct")

agent = create_agent(
    model=model,
    system_prompt="You are a helpful assistant and give precise answers. You don't give your thoughts just do what has been asked",
    tools=[get_weather],
)

app = FastAPI()


@app.get("/")
def ping():
    messages = [
        SystemMessage(
            content="You are a helpful assistant and give precise answers. You don't give your thoughts just do what has been asked"
        ),
        HumanMessage(content="what is the weather in India"),
    ]

    response = agent.invoke({"messages": messages})
    return {"ai_response": response}
