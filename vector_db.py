# vector_db.py
import threading
from typing import Any, Dict, Optional

from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient


class SingletonMeta(type):
    _instances: Dict[type, Any] = {}
    _locks: Dict[type, threading.Lock] = {}

    def __call__(cls, *args, **kwargs):
        if cls not in SingletonMeta._locks:
            SingletonMeta._locks[cls] = threading.Lock()
        if cls not in SingletonMeta._instances:
            with SingletonMeta._locks[cls]:
                if cls not in SingletonMeta._instances:
                    SingletonMeta._instances[cls] = super().__call__(*args, **kwargs)
        return SingletonMeta._instances[cls]


class VectorDB(metaclass=SingletonMeta):
    def __init__(self):
        # tiny, fast init only — nothing blocking here.
        self._client: Optional[QdrantClient] = None
        self._vector_store: Optional[QdrantVectorStore] = None
        self._init_lock = threading.Lock()

    def _ensure_initialized(self):
        # called lazily — safe to be called from multiple threads
        if self._client is not None and self._vector_store is not None:
            return

        with self._init_lock:
            if self._client is None:
                # create client here (fast), or add retries/timeouts if needed
                self._client = QdrantClient(url="http://host.docker.internal:6333")

            if self._vector_store is None:
                embedding = OllamaEmbeddings(model="bge-m3:latest")
                # Note: QdrantVectorStore constructor might check/create collection or do I/O
                # If it does, consider creating it in startup (Option B).
                self._vector_store = QdrantVectorStore(
                    client=self._client,
                    collection_name="test",
                    embedding=embedding,
                )

    @property
    def client(self) -> QdrantClient:
        self._ensure_initialized()
        assert self._client is not None
        return self._client

    @property
    def vector_store(self) -> QdrantVectorStore:
        self._ensure_initialized()
        assert self._vector_store is not None
        return self._vector_store

    # convenience wrapper
    def upload_vectors(self, *args, **kwargs):
        return self.client.upsert(*args, **kwargs)


# module-level instance is fine — it won't block now
vector_db = VectorDB()
