"""SigLIP-based image and text embeddings for multimodal retrieval (same vector space)."""

from __future__ import annotations

import io
from typing import List, Union

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel

from app.core import logger, settings

# Lazy-loaded singleton
_model = None
_processor = None


def _get_model():
    global _model, _processor
    if _model is None:
        model_name = settings.image_embedding_model
        logger.info("Loading image embedding model: %s", model_name)
        _processor = AutoProcessor.from_pretrained(model_name)
        _model = AutoModel.from_pretrained(model_name)
        if torch.cuda.is_available():
            _model = _model.cuda()
        _model.eval()
    return _model, _processor


def _to_device(inputs: dict, device) -> dict:
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}


def embed_image(image: Union[bytes, Image.Image]) -> List[float]:
    """Embed a single image; returns a list of floats (normalized, same space as text)."""
    if isinstance(image, bytes):
        image = Image.open(io.BytesIO(image)).convert("RGB")
    elif not isinstance(image, Image.Image):
        raise TypeError("image must be bytes or PIL.Image")

    model, processor = _get_model()
    device = next(model.parameters()).device
    # Pass dummy text so processor returns input_ids; full forward then gives projected image_embeds
    inputs = processor(
        text=["a photo"],
        images=image,
        padding="max_length",
        max_length=64,
        return_tensors="pt",
    )
    inputs = _to_device(inputs, device)

    with torch.no_grad():
        outputs = model(**inputs)
    emb = outputs.image_embeds
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb[0].cpu().tolist()


def embed_query_text(query: str) -> List[float]:
    """Embed a text query for searching the image vector space (SigLIP text encoder)."""
    model, processor = _get_model()
    device = next(model.parameters()).device
    inputs = processor(
        text=[query],
        padding="max_length",
        max_length=64,
        truncation=True,
        return_tensors="pt",
    )
    inputs = _to_device(inputs, device)

    with torch.no_grad():
        # Forward with only text to get projected text_embeds
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
        )
    emb = outputs.text_embeds
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb[0].cpu().tolist()


def image_embedding_dim() -> int:
    """Return the dimension of the image (and query text) embedding."""
    model, _ = _get_model()
    # Projection size for image_embeds / text_embeds (shared space)
    return getattr(
        model.config,
        "projection_dim",
        model.config.vision_config.hidden_size,
    )
