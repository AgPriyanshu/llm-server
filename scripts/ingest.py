"""Data ingestion script for loading documents into the vector database."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.models import Distance, VectorParams

from app.core import logger, settings
from app.services.vector_db import vector_db


class DataIngestionPipeline:
    """Pipeline for ingesting documents into the vector database."""

    def __init__(self, document_path: str):
        self.document_path = Path(document_path)
        # Local HuggingFace embeddings - no external server needed
        self.embeddings_model = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True},
        )

    def load_documents(self):
        """Load documents from PDF file."""
        loader = PyPDFLoader(self.document_path)
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} pages from {self.document_path}")
        return documents

    def run(self):
        """Execute the ingestion pipeline."""
        docs = self.load_documents()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True,
        )
        all_splits = text_splitter.split_documents(docs)
        logger.info(f"Split into {len(all_splits)} chunks")

        vector_size = len(self.embeddings_model.embed_query("sample text"))
        client = vector_db.client

        if not client.collection_exists(settings.qdrant_collection):
            client.create_collection(
                collection_name=settings.qdrant_collection,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            logger.info(f"Created collection: {settings.qdrant_collection}")

        vector_store = QdrantVectorStore(
            client=client,
            collection_name=settings.qdrant_collection,
            embedding=self.embeddings_model,
        )

        vector_store.add_documents(documents=all_splits)
        logger.info(f"Added {len(all_splits)} documents to vector store")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest documents into vector DB")
    parser.add_argument(
        "--document",
        "-d",
        type=str,
        default="./documents/how_mining_works.pdf",
        help="Path to the document to ingest",
    )
    args = parser.parse_args()

    pipeline = DataIngestionPipeline(document_path=args.document)
    pipeline.run()

