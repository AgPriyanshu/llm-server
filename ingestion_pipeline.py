from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables import chain
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.models import Distance, VectorParams

from vector_db import vector_db


class DataIngestionPipeline:
    def __init__(self, document_path: str):
        self.document_path = Path(document_path)
        self.embeddings_model = OllamaEmbeddings(model="bge-m3:latest")

    def load_documents(self):
        loader = PyPDFLoader(self.document_path)
        documents = loader.load()
        return documents

    def run(self):
        docs = self.load_documents()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, add_start_index=True
        )
        all_splits = text_splitter.split_documents(docs)

        vector_size = len(self.embeddings_model.embed_query("sample text"))
        client = vector_db.client

        if not client.collection_exists("test"):
            client.create_collection(
                collection_name="test",
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

        vector_store = QdrantVectorStore(
            client=client,
            collection_name="test",
            embedding=self.embeddings_model,
        )

        vector_store.add_documents(documents=all_splits)


if __name__ == "__main__":
    dip = DataIngestionPipeline(document_path="./documents/how_mining_works.pdf")
    dip.run()
