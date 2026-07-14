"""Data ingestion script for loading documents into the vector database."""

import base64
import io
import logging
import os
import sys
import uuid
from pathlib import Path

# Add parent directory to path for imports
_script_dir = Path(__file__).resolve().parent
_app_root = _script_dir.parent
sys.path.insert(0, str(_app_root))

# Use a single cache dir so models are not re-downloaded (override with HF_HOME / MODEL_CACHE_DIR)
_cache = _app_root / ".cache" / "models"
_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(_cache))
os.environ.setdefault("TRANSFORMERS_CACHE", str(_cache))
os.environ.setdefault("HF_HUB_CACHE", str(_cache / "hub"))

# Reduce noise from PDF parsing (pdfminer emits many "stroke color" warnings per page)
logging.getLogger("pdfminer.pdfinterp").setLevel(logging.ERROR)
logging.getLogger("pdfminer.pdfpage").setLevel(logging.ERROR)

from langchain_core.documents import Document as LangChainDocument
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core import logger, settings
from app.services.image_embedding import embed_image, image_embedding_dim

# Docling for PDF extraction (tables, images, text); fallback to PyPDF if Docling unavailable
try:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling_core.types.doc import PictureItem, TableItem

    HAS_DOCLING = True
except ImportError:
    HAS_DOCLING = False

from langchain_community.document_loaders import PyPDFLoader

# Element categories (aligned with Docling output; used for chunking and payload)
TABLE_LIKE_CATEGORIES = {"Table", "Table narrative"}
TEXT_CATEGORIES = {
    "NarrativeText",
    "Title",
    "Header",
    "ListItem",
    "Text",
    "UncategorizedText",
}
IMAGE_CATEGORIES = {"Image"}


def _page_no_from_element(element) -> int | None:
    """Get 1-based page number from docling element provenance if available."""
    prov = getattr(element, "prov", None)
    if prov is None:
        return None
    pno = getattr(prov, "page_no", None)
    return int(pno) if pno is not None else None


def _load_pdf_docling(
    document_path: Path, extract_images: bool = True
) -> list[LangChainDocument]:
    """Load PDF with Docling: text, tables, and optionally pictures as LangChain-style Documents."""
    pipeline_options = PdfPipelineOptions()
    pipeline_options.generate_picture_images = extract_images
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )
    conv_res = converter.convert(str(document_path))
    doc = conv_res.document
    source = str(document_path)
    out: list[LangChainDocument] = []

    for element, _level in doc.iterate_items():
        page_no = _page_no_from_element(element)

        if isinstance(element, TableItem):
            try:
                import pandas as pd

                df = element.export_to_dataframe(doc=doc)
                table_text = (
                    df.to_markdown()
                    if hasattr(df, "to_markdown")
                    else df.to_csv(index=False)
                )
            except Exception as e:
                logger.warning("Docling table export failed: %s", e)
                table_text = "[Table content not extracted]"
            out.append(
                LangChainDocument(
                    page_content=table_text,
                    metadata={
                        "source": source,
                        "page_number": page_no,
                        "category": "Table",
                    },
                )
            )
            continue

        if isinstance(element, PictureItem) and extract_images:
            try:
                img = element.get_image(doc)
                buf = io.BytesIO()
                img.save(buf, "PNG")
                image_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            except Exception as e:
                logger.warning("Docling picture export failed: %s", e)
                image_b64 = None
            out.append(
                LangChainDocument(
                    page_content="",
                    metadata={
                        "source": source,
                        "page_number": page_no,
                        "category": "Image",
                        **({"image_base64": image_b64} if image_b64 else {}),
                    },
                )
            )
            continue

        # Text and other items
        text = getattr(element, "text", None) or ""
        if isinstance(text, str) and text.strip():
            out.append(
                LangChainDocument(
                    page_content=text.strip(),
                    metadata={
                        "source": source,
                        "page_number": page_no,
                        "category": "Text",
                    },
                )
            )

    return out


def _load_pdf_fallback(document_path: Path) -> list:
    """Fallback: load PDF as plain pages with PyPDF (no table/image structure)."""
    loader = PyPDFLoader(str(document_path))
    return loader.load()


def _get_category(doc) -> str:
    """Return element category from document metadata, or 'Text' for legacy loaders."""
    meta = getattr(doc, "metadata", None) or {}
    return (meta.get("category") or "Text").strip()


def _chunk_documents_structure_aware(docs, text_splitter, table_max_chars: int = 8000):
    """
    Chunk documents by element type for best retrieval on mixed PDFs (tables, images, text).
    - Tables: kept as single chunks (or split by rows if very large).
    - Images: single chunk (caption/metadata only; no image embedding).
    - Text: split by paragraphs then sentences with overlap.
    """
    result = []
    for doc in docs:
        category = _get_category(doc)
        text = (doc.page_content or "").strip()

        if not text and category not in IMAGE_CATEGORIES:
            continue

        # Tables: keep as one chunk so queries can match full table context
        if category in TABLE_LIKE_CATEGORIES:
            if len(text) <= table_max_chars:
                result.append(doc)
            else:
                # Very large table: split by double newline (e.g. row groups) or fixed size
                for sub in text_splitter.split_documents([doc]):
                    result.append(sub)
            continue

        # Images: one chunk per image (text is often caption or empty)
        if category in IMAGE_CATEGORIES:
            result.append(doc)
            continue

        # Text elements: semantic split (paragraphs, then sentences)
        if category in TEXT_CATEGORIES or category not in (
            TABLE_LIKE_CATEGORIES | IMAGE_CATEGORIES
        ):
            splits = text_splitter.split_documents([doc])
            result.extend(splits)
            continue

        # Unknown category: treat as text
        splits = text_splitter.split_documents([doc])
        result.extend(splits)

    return result


def _split_text_and_image_chunks(chunks):
    """Split chunks into text/table (for BGE-M3) and image (for SigLIP)."""
    text_chunks = []
    image_chunks = []
    for doc in chunks:
        category = _get_category(doc)
        if category in IMAGE_CATEGORIES:
            image_chunks.append(doc)
        else:
            text_chunks.append(doc)
    return text_chunks, image_chunks


# Cache directory for HuggingFace models
MODEL_CACHE_DIR = Path(settings.model_cache_dir)


class DataIngestionPipeline:
    """Pipeline for ingesting documents into the vector database (Docling for PDF)."""

    def __init__(self, document_path: str, use_docling: bool = True):
        self.document_path = Path(document_path)
        self.use_docling = use_docling and HAS_DOCLING
        if use_docling and not HAS_DOCLING:
            logger.warning(
                "Docling not installed; install with: pip install docling. "
                "Falling back to PyPDF (no table/image structure)."
            )
        self.embeddings_model = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            cache_folder=str(MODEL_CACHE_DIR),
            model_kwargs={"device": settings.embedding_device},
            encode_kwargs={"normalize_embeddings": True},
        )

    def load_documents(self, extract_images: bool = True):
        """Load documents from PDF (Docling: tables, images, text; or PyPDF fallback)."""
        if self.use_docling:
            documents = _load_pdf_docling(
                self.document_path,
                extract_images=extract_images,
            )
            logger.info(
                "Loaded %s elements from %s (Docling: tables/images/text)",
                len(documents),
                self.document_path,
            )
        else:
            documents = _load_pdf_fallback(self.document_path)
            logger.info(
                "Loaded %s pages from %s (PyPDF)", len(documents), self.document_path
            )
        return documents

    def run(self):
        """Execute the ingestion pipeline with structure-aware chunking and image embeddings."""
        docs = self.load_documents()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
            add_start_index=True,
        )

        if self.use_docling:
            all_splits = _chunk_documents_structure_aware(docs, text_splitter)
            text_chunks, image_chunks = _split_text_and_image_chunks(all_splits)
        else:
            all_splits = text_splitter.split_documents(docs)
            text_chunks = all_splits
            image_chunks = []

        logger.info(
            "Split into %s chunks (%s text/table, %s image)",
            len(all_splits),
            len(text_chunks),
            len(image_chunks),
        )

        text_dim = len(self.embeddings_model.embed_query("sample text"))
        image_dim = image_embedding_dim()
        table_name = settings.pg_vector_table

        import psycopg
        from pgvector.psycopg import register_vector

        conn = psycopg.connect(settings.pg_vector_url)
        try:
            register_vector(conn)
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        id UUID PRIMARY KEY,
                        source TEXT,
                        page_number INT,
                        content_type TEXT,
                        page_content TEXT,
                        text_embedding vector({text_dim}),
                        image_embedding vector({image_dim})
                    )
                    """
                )
            conn.commit()
            logger.info(
                "Table %s ready (text_dim=%s, image_dim=%s)",
                table_name,
                text_dim,
                image_dim,
            )
        finally:
            conn.close()

        def _row(
            doc, content_type: str, page_content: str, text_emb=None, image_emb=None
        ):
            m = doc.metadata or {}
            return (
                uuid.uuid4(),
                m.get("source", str(self.document_path)),
                m.get("page_number"),
                content_type,
                page_content or "",
                text_emb,
                image_emb,
            )

        rows = []

        # Text/table chunks: embed with BGE-M3
        if text_chunks:
            text_embeddings = self.embeddings_model.embed_documents(
                [d.page_content for d in text_chunks]
            )
            for doc, emb in zip(text_chunks, text_embeddings):
                ct = "table" if _get_category(doc) in TABLE_LIKE_CATEGORIES else "text"
                rows.append(
                    _row(doc, ct, doc.page_content or "", text_emb=emb, image_emb=None)
                )

        # Image chunks: embed with SigLIP (no image_base64 stored)
        for doc in image_chunks:
            meta = doc.metadata or {}
            image_b64 = meta.get("image_base64")
            if not image_b64:
                logger.warning("Image chunk has no image_base64; skipping")
                continue
            try:
                image_bytes = base64.b64decode(image_b64)
                emb = embed_image(image_bytes)
            except Exception as e:
                logger.warning("Failed to embed image: %s", e)
                continue
            rows.append(
                _row(doc, "image", doc.page_content or "", text_emb=None, image_emb=emb)
            )

        if rows:
            conn = psycopg.connect(settings.pg_vector_url)
            try:
                register_vector(conn)
                batch_size = 200
                with conn.cursor() as cur:
                    for i in range(0, len(rows), batch_size):
                        batch = rows[i : i + batch_size]
                        cur.executemany(
                            f"""
                            INSERT INTO {table_name}
                            (id, source, page_number, content_type, page_content, text_embedding, image_embedding)
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                            """,
                            batch,
                        )
                conn.commit()
                # Create ivfflat indexes after data is loaded (cosine similarity for RAG)
                with conn.cursor() as cur:
                    cur.execute(
                        f"""
                        CREATE INDEX IF NOT EXISTS idx_{table_name}_text
                        ON {table_name} USING ivfflat (text_embedding vector_cosine_ops)
                        WITH (lists = 100)
                        """
                    )
                    cur.execute(
                        f"""
                        CREATE INDEX IF NOT EXISTS idx_{table_name}_image
                        ON {table_name} USING ivfflat (image_embedding vector_cosine_ops)
                        WITH (lists = 100)
                        """
                    )
                conn.commit()
                logger.info(
                    "Upserted %s rows to pgvector table %s", len(rows), table_name
                )
            finally:
                conn.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingest documents into pgvector (PostgreSQL). Uses Docling for PDF."
    )
    parser.add_argument(
        "--document",
        "-d",
        type=str,
        default="../documents/how_mining_works.pdf",
        help="Path to the document to ingest",
    )
    parser.add_argument(
        "--no-docling",
        action="store_true",
        help="Use PyPDF only (no table/image extraction)",
    )
    args = parser.parse_args()

    pipeline = DataIngestionPipeline(
        document_path=args.document,
        use_docling=not args.no_docling,
    )
    pipeline.run()
