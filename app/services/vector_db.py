"""Vector database service using Qdrant."""

import threading
from typing import Any, List, Optional
from pathlib import Path
from typing import Any, Optional

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from app.core import logger, settings
from app.services.image_embedding import embed_query_text as image_embed_query_text

# RRF constant (reciprocal rank fusion)
RRF_K = 60


def _rrf_merge(
    text_hits: List[Any],
    image_hits: List[Any],
    text_weight: float = 1.0,
    image_weight: float = 1.0,
) -> List[tuple[Any, float]]:
    """Merge two hit lists by Reciprocal Rank Fusion. Returns (hit, rrf_score) sorted by score."""
    scores: dict[Any, float] = {}
    for rank, hit in enumerate(text_hits):
        point_id = hit.id
        scores[point_id] = scores.get(point_id, 0.0) + text_weight / (RRF_K + rank + 1)
    for rank, hit in enumerate(image_hits):
        point_id = hit.id
        scores[point_id] = scores.get(point_id, 0.0) + image_weight / (RRF_K + rank + 1)
    # Rebuild list: all hits from both, sorted by rrf score
    by_id = {h.id: h for h in text_hits}
    by_id.update({h.id: h for h in image_hits})
    merged = [
        (by_id[pid], scores[pid])
        for pid in sorted(scores, key=lambda x: scores[x], reverse=True)
    ]
    return merged

# Model cache directory - persists downloaded models
MODEL_CACHE_DIR = Path(settings.model_cache_dir)


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
        self._text_embedding: Optional[HuggingFaceEmbeddings] = None
        self._init_lock = threading.Lock()

    def _ensure_initialized(self):
        """Initialize the database connection and text embedding lazily."""
        if self._client is None:
            with self._init_lock:
                if self._client is None:
                    try:
                        self._client = QdrantClient(url=settings.qdrant_url)
                        logger.info("Connected to Qdrant at %s", settings.qdrant_url)
                    except Exception as exc:
                        logger.error("Failed to connect to Qdrant: %s", exc)
                        raise

        if self._text_embedding is None:
            with self._init_lock:
                if self._text_embedding is None:
                    self._text_embedding = HuggingFaceEmbeddings(

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
                    # Ensure cache directory exists
                    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Loading embedding model (cache: {MODEL_CACHE_DIR})")

                    embedding = HuggingFaceEmbeddings(
                        model_name=settings.embedding_model,
                        cache_folder=str(MODEL_CACHE_DIR),
                        model_kwargs={"device": settings.embedding_device},
                        encode_kwargs={"normalize_embeddings": True},
                    )

        if self._vector_store is None and self._client.collection_exists(
            settings.qdrant_collection
        ):
            with self._init_lock:
                if self._vector_store is None and self._client.collection_exists(
                    settings.qdrant_collection
                ):
                    try:
                        self._vector_store = QdrantVectorStore(
                            client=self._client,
                            collection_name=settings.qdrant_collection,
                            embedding=self._text_embedding,
                            vector_name="text",
                        )
                        logger.info(
                            "Vector store initialized: %s",
                            settings.qdrant_collection,
                        )
                    except Exception as exc:
                        logger.error("Failed to initialize vector store: %s", exc)
                        raise

    @property
    def client(self) -> QdrantClient:
        """Get the Qdrant client."""
        self._ensure_initialized()
        assert self._client is not None
        return self._client

    @property
    def vector_store(self) -> QdrantVectorStore:
        """Get the vector store (for single-vector use; prefer similarity_search for multimodal)."""
        self._ensure_initialized()
        assert self._vector_store is not None
        return self._vector_store

    def similarity_search(
        self,
        query: str,
        k: int = 6,
    ) -> List[Document]:
        """
        Multimodal similarity search: query both text and image vectors, merge with RRF,
        return LangChain Documents with content_type in metadata.
        Falls back to single-vector search if the collection has no named vectors.
        """
        self._ensure_initialized()
        if not self._client.collection_exists(settings.qdrant_collection):
            return []

        try:
            text_weight = 1.0 - settings.retrieval_image_weight
            image_weight = settings.retrieval_image_weight

            query_text_emb = self._text_embedding.embed_query(query)
            query_image_emb = image_embed_query_text(query)

            text_hits = self._client.search(
                collection_name=settings.qdrant_collection,
                query_vector=("text", query_text_emb),
                limit=k,
                with_payload=True,
            )
            image_hits = self._client.search(
                collection_name=settings.qdrant_collection,
                query_vector=("image", query_image_emb),
                limit=k,
                with_payload=True,
            )

            merged = _rrf_merge(text_hits, image_hits, text_weight, image_weight)
            docs: List[Document] = []
            seen_ids: set[Any] = set()
            for hit, _ in merged[:k]:
                if hit.id in seen_ids:
                    continue
                seen_ids.add(hit.id)
                payload = hit.payload or {}
                page_content = payload.get("page_content", "")
                metadata = {k: v for k, v in payload.items() if k != "image_base64"}
                docs.append(Document(page_content=page_content, metadata=metadata))
            return docs
        except Exception as e:
            # Collection may have single (unnamed) vector from before multimodal ingest
            logger.debug("Dual-vector search failed, falling back to single-vector: %s", e)
            if self._vector_store is not None:
                return self._vector_store.similarity_search(query, k=k)
            # Last resort: search with text vector on default name
            query_text_emb = self._text_embedding.embed_query(query)
            hits = self._client.search(
                collection_name=settings.qdrant_collection,
                query_vector=query_text_emb,
                limit=k,
                with_payload=True,
            )
            return [
                Document(
                    page_content=(h.payload or {}).get("page_content", ""),
                    metadata={k: v for k, v in (h.payload or {}).items() if k != "image_base64"},
                )
                for h in hits
            ]

    def upload_vectors(self, *args, **kwargs):
        """Convenience wrapper for upserting vectors."""
        return self.client.upsert(*args, **kwargs)


# Module-level singleton instance
vector_db = VectorDB()

