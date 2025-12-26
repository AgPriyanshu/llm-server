import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import chat, health
from app.api.websocket import router as ws_router
from app.core import logger, settings
from app.services.vector_db import VectorDB


async def init_vector_db() -> None:
    """Initialize VectorDB in a background thread."""
    loop = asyncio.get_running_loop()
    db = VectorDB()

    def sync_init():
        db._ensure_initialized()
        return True

    await loop.run_in_executor(None, sync_init)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown."""
    # Startup
    logger.info("Starting application...")
    try:
        await init_vector_db()
        logger.info("VectorDB initialized successfully")
    except Exception as exc:
        logger.error(f"Failed to initialize VectorDB: {exc}")
        raise

    yield

    # Shutdown
    logger.info("Application shutting down")


def create_app() -> FastAPI:
    """Application factory for creating the FastAPI app."""
    app = FastAPI(
        title=settings.app_name,
        description="Production-ready LLM server with RAG capabilities",
        version=settings.app_version,
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(ws_router)

    return app


# Create the app instance
app = create_app()

