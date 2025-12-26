"""Vector database service using Qdrant."""

import threading
from typing import Any, Optional

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from app.core import logger, settings


class SingletonMeta(type):
    """Thread-safe singleton metaclass."""

    _instances: dict[type, Any] = {}
    _locks: dict[type, threading.Lock] = {}

    def __call__(cls, *args, **kwargs):
        if cls not in SingletonMeta._locks:
            SingletonMeta._locks[cls] = threading.Lock()
        if cls not in SingletonMeta._instances:
            with SingletonMeta._locks[cls]:
                if cls not in SingletonMeta._instances:
                    SingletonMeta._instances[cls] = super().__call__(*args, **kwargs)
        return SingletonMeta._instances[cls]


class VectorDB(metaclass=SingletonMeta):
    """Singleton vector database service."""

    def __init__(self):
        self._client: Optional[QdrantClient] = None
        self._vector_store: Optional[QdrantVectorStore] = None
        self._init_lock = threading.Lock()

    def _ensure_initialized(self):
        """Initialize the database connection lazily."""
        if self._client is not None and self._vector_store is not None:
            return

        with self._init_lock:
            if self._client is None:
                try:
                    self._client = QdrantClient(url=settings.qdrant_url)
                    logger.info(f"Connected to Qdrant at {settings.qdrant_url}")
                except Exception as exc:
                    logger.error(f"Failed to connect to Qdrant: {exc}")
                    raise

            if self._vector_store is None and self._client.collection_exists(
                settings.qdrant_collection
            ):
                try:
                    embedding = HuggingFaceEmbeddings(
                        model_name=settings.embedding_model,
                        model_kwargs={"device": "cuda"},
                        encode_kwargs={"normalize_embeddings": True},
                    )
                    self._vector_store = QdrantVectorStore(
                        client=self._client,
                        collection_name=settings.qdrant_collection,
                        embedding=embedding,
                    )
                    logger.info(
                        f"Vector store initialized: {settings.qdrant_collection}"
                    )
                except Exception as exc:
                    logger.error(f"Failed to initialize vector store: {exc}")
                    raise

    @property
    def client(self) -> QdrantClient:
        """Get the Qdrant client."""
        self._ensure_initialized()
        assert self._client is not None
        return self._client

    @property
    def vector_store(self) -> QdrantVectorStore:
        """Get the vector store."""
        self._ensure_initialized()
        assert self._vector_store is not None
        return self._vector_store

    def upload_vectors(self, *args, **kwargs):
        """Convenience wrapper for upserting vectors."""
        return self.client.upsert(*args, **kwargs)


# Module-level singleton instance
vector_db = VectorDB()

