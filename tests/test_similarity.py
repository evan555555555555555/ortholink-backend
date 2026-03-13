"""
Tests for similarity tool.
"""

import numpy as np

from app.tools.similarity import cosine_similarity, semantic_match


class TestCosineSimilarity:
    """Test cosine similarity computation."""

    def test_identical_vectors(self):
        v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        sim = cosine_similarity(v, v)
        assert abs(sim - 1.0) < 1e-5

    def test_orthogonal_vectors(self):
        v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        sim = cosine_similarity(v1, v2)
        assert abs(sim) < 1e-5

    def test_opposite_vectors(self):
        v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
        sim = cosine_similarity(v1, v2)
        assert abs(sim - (-1.0)) < 1e-5

    def test_similar_vectors(self):
        v1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        v2 = np.array([1.1, 2.1, 3.1], dtype=np.float32)
        sim = cosine_similarity(v1, v2)
        assert sim > 0.99

    def test_returns_float(self):
        v1 = np.array([1.0, 2.0], dtype=np.float32)
        v2 = np.array([3.0, 4.0], dtype=np.float32)
        sim = cosine_similarity(v1, v2)
        assert isinstance(sim, float)


class TestSemanticMatch:
    """Test semantic matching with 0.82 threshold."""

    def test_high_similarity_matches(self):
        v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = semantic_match(v, v, threshold=0.82)
        assert result["is_match"] is True
        assert result["similarity"] >= 0.82

    def test_low_similarity_no_match(self):
        v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        result = semantic_match(v1, v2, threshold=0.82)
        assert result["is_match"] is False
        assert result["similarity"] < 0.82

    def test_default_threshold_082(self):
        v = np.array([1.0, 0.0], dtype=np.float32)
        result = semantic_match(v, v)
        assert result["threshold"] == 0.82

    def test_custom_threshold(self):
        v = np.array([1.0, 0.0], dtype=np.float32)
        result = semantic_match(v, v, threshold=0.95)
        assert result["threshold"] == 0.95

    def test_result_has_strength(self):
        v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = semantic_match(v, v)
        assert "strength" in result
