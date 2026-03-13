"""
OrthoLink Embeddings Tool
HC-1: text-embedding-3-large (3072 dims) ONLY. No other model permitted.
"""

import logging
from typing import Optional

import numpy as np
from openai import OpenAI

from app.core.config import get_settings

logger = logging.getLogger(__name__)

_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    """Lazy-initialize OpenAI client."""
    global _client
    if _client is None:
        settings = get_settings()
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client


def embed_text(text: str) -> np.ndarray:
    """
    Embed a single text string using text-embedding-3-large.
    Returns numpy array of shape (3072,).

    HC-1 ENFORCED: Only text-embedding-3-large is used.
    """
    settings = get_settings()
    client = _get_client()

    response = client.embeddings.create(
        model=settings.openai_embedding_model,
        input=text,
        dimensions=settings.openai_embedding_dimensions,
    )

    embedding = response.data[0].embedding
    return np.array(embedding, dtype=np.float32)


def embed_batch(texts: list[str], batch_size: int = 100) -> np.ndarray:
    """
    Embed a batch of texts using text-embedding-3-large.
    Returns numpy array of shape (n, 3072).

    Processes in batches to respect API limits.
    HC-1 ENFORCED: Only text-embedding-3-large is used.
    """
    settings = get_settings()
    client = _get_client()
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(
            model=settings.openai_embedding_model,
            input=batch,
            dimensions=settings.openai_embedding_dimensions,
        )
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)
        logger.info(f"Embedded batch {i // batch_size + 1}, total: {len(all_embeddings)}/{len(texts)}")

    return np.array(all_embeddings, dtype=np.float32)
