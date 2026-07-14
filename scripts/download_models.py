"""Pre-download and cache all model weights used by the ingest and retrieval pipeline.

Run once (e.g. in Docker build or before first ingest) so later runs use the cache:
  python scripts/download_models.py

Uses HF_HOME / TRANSFORMERS_CACHE (default: llm_server/.cache/models).
"""

import os
import sys
import tempfile
from pathlib import Path

# Set cache before any model imports
_script_dir = Path(__file__).resolve().parent
_app_root = _script_dir.parent
sys.path.insert(0, str(_app_root))

_cache = _app_root / ".cache" / "models"
_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(_cache))
os.environ.setdefault("TRANSFORMERS_CACHE", str(_cache))
os.environ.setdefault("HF_HUB_CACHE", str(_cache / "hub"))

# Now safe to import app and model libs
from app.core import logger, settings


def download_text_embedding():
    """Download and cache BGE-M3 (text embedding)."""
    logger.info("Downloading text embedding model: %s", settings.embedding_model)
    from langchain_huggingface import HuggingFaceEmbeddings

    HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    ).embed_query("warmup")
    logger.info("Text embedding model cached.")


def download_image_embedding():
    """Download and cache SigLIP (image embedding). Requires sentencepiece (pip install sentencepiece)."""
    logger.info("Downloading image embedding model: %s", settings.image_embedding_model)
    try:
        from transformers import AutoModel, AutoProcessor

        AutoProcessor.from_pretrained(settings.image_embedding_model)
        AutoModel.from_pretrained(settings.image_embedding_model)
        logger.info("Image embedding model cached.")
    except ImportError as e:
        if "sentencepiece" in str(e).lower() or "SentencePiece" in str(e):
            logger.error(
                "SigLIP requires sentencepiece. Install with: pip install sentencepiece"
            )
        raise


def download_unstructured_table_model():
    """Trigger Unstructured table-structure model download (used for hi_res + infer_table_structure)."""
    try:
        from pypdf import PdfWriter

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            path = f.name
        try:
            w = PdfWriter()
            w.add_blank_page(width=72, height=72)
            with open(path, "wb") as f:
                w.write(f)
            logger.info("Downloading Unstructured table model (hi_res)...")
            from unstructured.partition.pdf import partition_pdf

            partition_pdf(path, strategy="hi_res", infer_table_structure=True)
            logger.info("Unstructured table model cached.")
        finally:
            Path(path).unlink(missing_ok=True)
    except Exception as e:
        logger.warning("Could not pre-cache Unstructured table model: %s", e)
        logger.info("It will download on first ingest with hi_res.")


def main():
    logger.info("Model cache directory: %s", _cache)
    download_text_embedding()
    download_image_embedding()
    download_unstructured_table_model()
    logger.info("All models cached. Subsequent ingest/retrieval will use the cache.")


if __name__ == "__main__":
    main()
