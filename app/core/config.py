"""Application configuration using pydantic-settings."""

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Default cache under package root so models are reused and can be mounted in Docker
_PKG_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_MODEL_CACHE = _PKG_ROOT / ".cache" / "models"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        # Environment variables are injected directly into the container
        env_file=None,
        extra="ignore",
    )

    # Application
    app_name: str = "LLM Server"
    app_version: str = "1.0.0"
    debug: bool = False

    # Inference server configuration
    inference_server_url: str = "http://host.docker.internal:11434"
    model_name: str = "qwen3:4b-instruct"

    # Qdrant configuration (legacy)
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "test"

    # pgvector (PostgreSQL) – used by ingest and optional for retrieval.
    # Default: rainbow-api-engine db (postgis on port 5431).
    pg_vector_url: str = "postgresql://postgres:postgres@localhost:5431/postgres"
    pg_vector_table: str = "document_embeddings"

    # Embedding model (HuggingFace model name)
    embedding_model: str = "BAAI/bge-m3"

    # Image embedding model for multimodal retrieval (e.g. SigLIP)
    image_embedding_model: str = "google/siglip-base-patch16-224"

    # Weight for image results when merging with text retrieval (0 = text only, 1 = image only)
    retrieval_image_weight: float = 0.5

    # Model cache directory (HF_HOME / TRANSFORMERS_CACHE). Default: llm_server/.cache/models
    model_cache_dir: str = str(_DEFAULT_MODEL_CACHE)

    # CORS
    cors_origins: list[str] = ["*"]


settings = Settings()


def ensure_model_cache_env() -> None:
    """Set HF_HOME and TRANSFORMERS_CACHE so all model downloads use the same cache."""
    cache = Path(settings.model_cache_dir).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache))
    os.environ.setdefault("HF_HUB_CACHE", str(cache / "hub"))


# Set cache env at import so server and any script importing app use the same cache
ensure_model_cache_env()

