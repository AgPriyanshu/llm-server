from fastapi import FastAPI
from langchain.chat_models import init_chat_model

app = FastAPI()

model = init_chat_model(
    model="Qwen/Qwen3-4B-Instruct-2507",
    model_provider="openai",
    base_url="http://localhost:8001/",
    api_key="YOUR_API_KEY",
)


@app.post("/chat")
def ping():
    return {"ping": "pong"}
