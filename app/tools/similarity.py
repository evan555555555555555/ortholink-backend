"""
OrthoLink Similarity Tool
Cosine similarity computation for semantic matching.
"""

import numpy as np


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    Returns float in range [-1, 1].
    """
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def batch_cosine_similarity(query_vec: np.ndarray, corpus_vecs: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between a query vector and a corpus of vectors.
    Returns array of similarities of shape (n,).
    """
    query_norm = np.linalg.norm(query_vec)
    if query_norm == 0:
        return np.zeros(corpus_vecs.shape[0])

    corpus_norms = np.linalg.norm(corpus_vecs, axis=1)
    # Avoid division by zero
    corpus_norms = np.where(corpus_norms == 0, 1e-10, corpus_norms)

    similarities = np.dot(corpus_vecs, query_vec) / (corpus_norms * query_norm)
    return similarities


def semantic_match(
    text_a_embedding: np.ndarray,
    text_b_embedding: np.ndarray,
    threshold: float = 0.82,
) -> dict:
    """
    Determine if two texts are semantically equivalent.
    Used for DVA matching (e.g., "Bank Statement" vs "Proof of Financial Stability").

    Returns dict with similarity score and match determination.
    Threshold default 0.82 per PRD requirement.
    """
    similarity = cosine_similarity(text_a_embedding, text_b_embedding)

    return {
        "similarity": round(similarity, 4),
        "is_match": similarity >= threshold,
        "threshold": threshold,
        "strength": _classify_match_strength(similarity),
    }


def _classify_match_strength(similarity: float) -> str:
    """Classify the strength of a semantic match."""
    if similarity >= 0.95:
        return "exact"
    elif similarity >= 0.90:
        return "very_strong"
    elif similarity >= 0.85:
        return "strong"
    elif similarity >= 0.80:
        return "moderate"
    elif similarity >= 0.70:
        return "weak"
    else:
        return "no_match"
