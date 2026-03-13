"""
OrthoLink Embedder
Batch embedding + FAISS index builder.
HC-1: text-embedding-3-large (3072 dims) ONLY.
"""

import logging
from typing import Optional

from app.ingestion.chunker import Chunk
from app.tools.embeddings import embed_batch
from app.tools.vector_store import ChunkMetadata, VectorStore, get_vector_store

logger = logging.getLogger(__name__)


def embed_and_index_chunks(
    chunks: list[Chunk],
    vector_store: Optional[VectorStore] = None,
    batch_size: int = 100,
) -> int:
    """
    Embed a list of chunks and add them to the FAISS vector store.
    Returns the number of chunks successfully embedded and indexed.

    HC-1 ENFORCED: Uses text-embedding-3-large via embed_batch.
    """
    if not chunks:
        logger.warning("No chunks to embed")
        return 0

    store = vector_store or get_vector_store()

    # Prepare texts for embedding (truncate to stay under 8192-token API limit per batch)
    max_chars_per_chunk = 2000
    texts = [
        (chunk.text[:max_chars_per_chunk] if len(chunk.text) > max_chars_per_chunk else chunk.text)
        for chunk in chunks
    ]

    # Embed in batches
    logger.info(f"Embedding {len(texts)} chunks...")
    embeddings = embed_batch(texts, batch_size=batch_size)

    # Create metadata objects
    metadata_list = [
        ChunkMetadata(
            chunk_id=chunk.chunk_id,
            country=chunk.country,
            regulation_name=chunk.regulation_name,
            article=chunk.article,
            clause=chunk.clause,
            device_classes=chunk.device_classes,
            text=chunk.text,
            parent_text=chunk.parent_text,
            source_url=chunk.source_url,
            language=chunk.language,
            original_language=chunk.original_language,
            is_active=True,
            chunk_hash=chunk.chunk_hash,
            document_id=getattr(chunk, "document_id", None),
            section_path=getattr(chunk, "section_path", None),
        )
        for chunk in chunks
    ]

    # Add to index
    store.add_chunks(embeddings, metadata_list)

    # Persist to disk
    store.save()

    logger.info(f"Successfully embedded and indexed {len(chunks)} chunks")
    return len(chunks)


def re_embed_chunks(
    old_chunk_ids: list[str],
    new_chunks: list[Chunk],
    vector_store: Optional[VectorStore] = None,
) -> int:
    """
    Re-embed chunks after a regulation update.
    Marks old chunks as inactive (never deletes — HC-6) and adds new ones.
    """
    store = vector_store or get_vector_store()

    # Mark old chunks as inactive (NEVER delete — audit requirement)
    from datetime import datetime, timezone
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    for metadata in store.metadata:
        if metadata.chunk_id in old_chunk_ids:
            metadata.is_active = False
            metadata.valid_to = today

    # Embed and add new chunks
    count = embed_and_index_chunks(new_chunks, store)

    logger.info(
        f"Re-embedded: {len(old_chunk_ids)} old chunks deactivated, "
        f"{count} new chunks added"
    )

    return count
