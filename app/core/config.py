"""Application configuration using pydantic-settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


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

    # Qdrant configuration
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "test"

    # Embedding model (HuggingFace model name)
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "cuda"  # "cuda" or "cpu"
    model_cache_dir: str = (
        "./data/models"  # Relative path works both locally and in Docker
    )

    # CORS
    cors_origins: list[str] = ["*"]


settings = Settings()
