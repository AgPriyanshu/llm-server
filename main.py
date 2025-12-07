from typing import Union

from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def ping():
    return {"ping": "pong"}
