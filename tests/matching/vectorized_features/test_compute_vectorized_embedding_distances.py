"""Tests for compute_vectorized_embedding_distances function."""

import numpy as np
import pytest
import scipy.spatial.distance

from mouse_tracking.matching.vectorized_features import (
    compute_vectorized_embedding_distances,
)


class TestComputeVectorizedEmbeddingDistances:
    """Test basic functionality of compute_vectorized_embedding_distances."""

    def test_basic_embedding_distance(self, features_factory):
        """Test basic embedding distance computation."""
        # Create features with different embeddings
        embed_configs = [
            {"has_embedding": True, "dim": 4, "value": 1.0},  # All ones
            {"has_embedding": True, "dim": 4, "value": 0.5},  # All 0.5s
        ]

        features1 = features_factory(
            n_detections=1, embed_configs=[embed_configs[0]], seed=42
        )
        features2 = features_factory(
            n_detections=1, embed_configs=[embed_configs[1]], seed=42
        )

        result = compute_vectorized_embedding_distances(features1, features2)

        # Should be a 1x1 matrix
        assert result.shape == (1, 1)

        # Compute expected distance manually
        embed1 = np.ones(4)
        embed2 = np.full(4, 0.5)
        expected = scipy.spatial.distance.cdist([embed1], [embed2], metric="cosine")[
            0, 0
        ]
        expected = np.clip(expected, 0, 1.0 - 1e-8)

        np.testing.assert_allclose(result[0, 0], expected, rtol=1e-10)

    def test_identical_embeddings(self, features_factory):
        """Test distance between identical embeddings."""
        embed_configs = [{"has_embedding": True, "dim": 128, "value": 0.7}]

        features1 = features_factory(
            n_detections=1, embed_configs=embed_configs, seed=42
        )
        features2 = features_factory(
            n_detections=1, embed_configs=embed_configs, seed=42
        )

        result = compute_vectorized_embedding_distances(features1, features2)

        # Should be approximately 0 (may not be exactly 0 due to floating point)
        assert result.shape == (1, 1)
        assert result[0, 0] < 1e-10

    def test_orthogonal_embeddings(self, features_factory):
        """Test distance between orthogonal embeddings."""
        # Create orthogonal vectors
        embed1 = np.array([1.0, 0.0, 0.0, 0.0])
        embed2 = np.array([0.0, 1.0, 0.0, 0.0])

        # Create features with these specific embeddings
        features1 = features_factory(
            n_detections=1, embed_configs=[{"has_embedding": False}]
        )
        features2 = features_factory(
            n_detections=1, embed_configs=[{"has_embedding": False}]
        )

        # Manually set the embeddings
        features1.embeddings = np.array([embed1])
        features1.valid_embed_masks = np.array([True])
        features2.embeddings = np.array([embed2])
        features2.valid_embed_masks = np.array([True])

        result = compute_vectorized_embedding_distances(features1, features2)

        # Cosine distance between orthogonal vectors should be clipped to 1.0 - 1e-8
        assert result.shape == (1, 1)
        expected_clipped = 1.0 - 1e-8
        np.testing.assert_allclose(result[0, 0], expected_clipped, rtol=1e-10)

    def test_matrix_computation(self, features_factory):
        """Test distance matrix for multiple embeddings."""
        embed_configs = [
            {"has_embedding": True, "dim": 3, "value": None},  # Random
            {"has_embedding": True, "dim": 3, "value": None},  # Random
            {"has_embedding": True, "dim": 3, "value": None},  # Random
        ]

        features1 = features_factory(
            n_detections=2, embed_configs=embed_configs[:2], seed=42
        )
        features2 = features_factory(
            n_detections=3, embed_configs=embed_configs, seed=100
        )

        result = compute_vectorized_embedding_distances(features1, features2)

        # Should be 2x3 matrix
        assert result.shape == (2, 3)

        # Check that all distances are valid
        assert np.all(~np.isnan(result))
        assert np.all(result >= 0)
        assert np.all(result <= 1.0)

        # Verify specific elements manually
        expected_01 = scipy.spatial.distance.cdist(
            [features1.embeddings[0]], [features2.embeddings[1]], metric="cosine"
        )[0, 0]
        expected_01 = np.clip(expected_01, 0, 1.0 - 1e-8)
        np.testing.assert_allclose(result[0, 1], expected_01, rtol=1e-10)

    def test_consistency_with_original_method(
        self, detection_factory, features_factory
    ):
        """Test consistency with Detection.embed_distance method."""
        from mouse_tracking.matching.core import Detection

        # Create detections with known embeddings
        det1 = detection_factory(pose_idx=0, embed_dim=64, seed=42)
        det2 = detection_factory(pose_idx=1, embed_dim=64, seed=100)

        # Test original method
        original_dist = Detection.embed_distance(det1.embed, det2.embed)

        # Test vectorized method
        features1 = features_factory(
            n_detections=1, embed_configs=[{"has_embedding": False}]
        )
        features2 = features_factory(
            n_detections=1, embed_configs=[{"has_embedding": False}]
        )
        features1.detections = [det1]
        features1.embeddings = np.array([det1.embed])
        features1.valid_embed_masks = np.array([True])
        features2.detections = [det2]
        features2.embeddings = np.array([det2.embed])
        features2.valid_embed_masks = np.array([True])

        vectorized_dist = compute_vectorized_embedding_distances(features1, features2)

        # Should match exactly
        np.testing.assert_allclose(vectorized_dist[0, 0], original_dist, rtol=1e-15)


class TestComputeVectorizedEmbeddingDistancesEdgeCases:
    """Test edge cases and invalid input handling."""

    def test_empty_embeddings_both_sides(self, features_factory):
        """Test with empty embeddings on both sides."""
        # Create features with no embeddings - need configs for all detections
        embed_configs1 = [{"has_embedding": False}, {"has_embedding": False}]
        embed_configs2 = [
            {"has_embedding": False},
            {"has_embedding": False},
            {"has_embedding": False},
        ]

        features1 = features_factory(n_detections=2, embed_configs=embed_configs1)
        features2 = features_factory(n_detections=3, embed_configs=embed_configs2)

        result = compute_vectorized_embedding_distances(features1, features2)

        # Should return all NaN
        assert result.shape == (2, 3)
        assert np.all(np.isnan(result))

    def test_empty_embeddings_one_side(self, features_factory):
        """Test with empty embeddings on one side."""
        embed_configs_valid = [
            {"has_embedding": True, "dim": 64},
            {"has_embedding": True, "dim": 64},
        ]
        embed_configs_empty = [{"has_embedding": False}]

        features1 = features_factory(n_detections=2, embed_configs=embed_configs_valid)
        features2 = features_factory(n_detections=1, embed_configs=embed_configs_empty)

        result = compute_vectorized_embedding_distances(features1, features2)

        # Should return all NaN
        assert result.shape == (2, 1)
        assert np.all(np.isnan(result))

    def test_zero_embeddings(self, features_factory):
        """Test with zero embeddings (invalid)."""
        # Create features with explicit zero embeddings
        features1 = features_factory(
            n_detections=1, embed_configs=[{"has_embedding": False}]
        )
        features2 = features_factory(
            n_detections=1, embed_configs=[{"has_embedding": False}]
        )

        # Manually set zero embeddings
        features1.embeddings = np.zeros((1, 128))
        features1.valid_embed_masks = np.array([False])  # Should be invalid
        features2.embeddings = np.zeros((1, 128))
        features2.valid_embed_masks = np.array([False])  # Should be invalid

        result = compute_vectorized_embedding_distances(features1, features2)

        # Should return NaN for invalid embeddings
        assert result.shape == (1, 1)
        assert np.isnan(result[0, 0])

    def test_mixed_valid_invalid_embeddings(self, features_factory):
        """Test with mixed valid and invalid embeddings."""
        # Create some valid, some invalid embeddings
        features1 = features_factory(
            n_detections=2,
            embed_configs=[
                {"has_embedding": True, "dim": 32, "value": 0.5},  # Valid
                {"has_embedding": False},  # Invalid (will be zeros)
            ],
        )
        features2 = features_factory(
            n_detections=2,
            embed_configs=[
                {"has_embedding": False},  # Invalid (will be zeros)
                {"has_embedding": True, "dim": 32, "value": 0.8},  # Valid
            ],
        )

        result = compute_vectorized_embedding_distances(features1, features2)

        assert result.shape == (2, 2)

        # Only (0,1) should be valid (valid vs valid)
        assert np.isnan(result[0, 0])  # valid vs invalid
        assert not np.isnan(result[0, 1])  # valid vs valid
        assert np.isnan(result[1, 0])  # invalid vs invalid
        assert np.isnan(result[1, 1])  # invalid vs valid

        # Check the valid distance
        assert 0 <= result[0, 1] <= 1.0

    def test_no_detections(self, features_factory):
        """Test with no detections."""
        features1 = features_factory(n_detections=0)
        features2 = features_factory(n_detections=0)

        result = compute_vectorized_embedding_distances(features1, features2)

        # Should return empty matrix
        assert result.shape == (0, 0)

    def test_mismatched_dimensions_error(self, features_factory):
        """Test error handling for mismatched embedding dimensions."""
        # This should be handled by the VectorizedDetectionFeatures initialization
        # but let's test the direct case
        features1 = features_factory(
            n_detections=1, embed_configs=[{"has_embedding": False}]
        )
        features2 = features_factory(
            n_detections=1, embed_configs=[{"has_embedding": False}]
        )

        # Manually create mismatched dimensions
        features1.embeddings = np.random.random((1, 64))
        features1.valid_embed_masks = np.array([True])
        features2.embeddings = np.random.random((1, 128))  # Different dimension
        features2.valid_embed_masks = np.array([True])

        # This should raise an error from scipy
        with pytest.raises(ValueError):
            compute_vectorized_embedding_distances(features1, features2)

    def test_single_detection_each_side(self, features_factory):
        """Test with single detection on each side."""
        features1 = features_factory(
            n_detections=1, embed_configs=[{"has_embedding": True, "dim": 16}]
        )
        features2 = features_factory(
            n_detections=1, embed_configs=[{"has_embedding": True, "dim": 16}]
        )

        result = compute_vectorized_embedding_distances(features1, features2)

        assert result.shape == (1, 1)
        assert not np.isnan(result[0, 0])
        assert 0 <= result[0, 0] <= 1.0


class TestComputeVectorizedEmbeddingDistancesProperties:
    """Test mathematical properties and correctness."""

    def test_distance_symmetry(self, features_factory):
        """Test that distance matrix is symmetric for same features."""
        features = features_factory(
            n_detections=3,
            embed_configs=[
                {"has_embedding": True, "dim": 32},
                {"has_embedding": True, "dim": 32},
                {"has_embedding": True, "dim": 32},
            ],
            seed=42,
        )

        result = compute_vectorized_embedding_distances(features, features)

        # Should be symmetric
        assert result.shape == (3, 3)
        np.testing.assert_allclose(result, result.T, rtol=1e-10)

        # Diagonal should be approximately zero
        diagonal = np.diag(result)
        assert np.all(diagonal < 1e-10)

    def test_distance_bounds(self, features_factory):
        """Test that distances are bounded correctly."""
        features1 = features_factory(n_detections=5, seed=42)
        features2 = features_factory(n_detections=7, seed=100)

        result = compute_vectorized_embedding_distances(features1, features2)

        # All valid distances should be in [0, 1]
        valid_mask = ~np.isnan(result)
        valid_distances = result[valid_mask]

        if len(valid_distances) > 0:
            assert np.all(valid_distances >= 0)
            assert np.all(valid_distances <= 1.0)

    def test_clipping_behavior(self, features_factory):
        """Test the clipping behavior matches original implementation."""
        # Create features that might produce edge case distances
        features1 = features_factory(
            n_detections=1, embed_configs=[{"has_embedding": False}]
        )
        features2 = features_factory(
            n_detections=1, embed_configs=[{"has_embedding": False}]
        )

        # Create embeddings that would produce distance exactly 1.0
        embed1 = np.array([1.0, 0.0])
        embed2 = np.array([-1.0, 0.0])  # Opposite direction

        features1.embeddings = np.array([embed1])
        features1.valid_embed_masks = np.array([True])
        features2.embeddings = np.array([embed2])
        features2.valid_embed_masks = np.array([True])

        result = compute_vectorized_embedding_distances(features1, features2)

        # Should be clipped to slightly less than 1.0
        assert result.shape == (1, 1)
        assert result[0, 0] <= 1.0 - 1e-8

        # Verify this matches the original clipping
        expected = scipy.spatial.distance.cdist([embed1], [embed2], metric="cosine")[
            0, 0
        ]
        expected = np.clip(expected, 0, 1.0 - 1e-8)
        np.testing.assert_allclose(result[0, 0], expected, rtol=1e-15)

    def test_random_embedding_consistency(self, features_factory):
        """Test consistency with random embeddings."""
        np.random.seed(12345)
        n1, n2 = 4, 6
        embed_dim = 64

        # Generate random embeddings
        embeddings1 = np.random.random((n1, embed_dim))
        embeddings2 = np.random.random((n2, embed_dim))

        # Create features with these embeddings
        features1 = features_factory(
            n_detections=n1, embed_configs=[{"has_embedding": False}] * n1
        )
        features2 = features_factory(
            n_detections=n2, embed_configs=[{"has_embedding": False}] * n2
        )

        features1.embeddings = embeddings1
        features1.valid_embed_masks = np.ones(n1, dtype=bool)
        features2.embeddings = embeddings2
        features2.valid_embed_masks = np.ones(n2, dtype=bool)

        result = compute_vectorized_embedding_distances(features1, features2)

        # Compute expected using scipy directly
        expected = scipy.spatial.distance.cdist(
            embeddings1, embeddings2, metric="cosine"
        )
        expected = np.clip(expected, 0, 1.0 - 1e-8)

        # Should match exactly
        np.testing.assert_allclose(result, expected, rtol=1e-15)


class TestComputeVectorizedEmbeddingDistancesPerformance:
    """Test performance characteristics."""

    def test_large_matrix_computation(self, features_factory):
        """Test computation with larger matrices."""
        # Test with moderately large matrices
        n1, n2 = 50, 60
        embed_dim = 256

        features1 = features_factory(
            n_detections=n1,
            embed_configs=[
                {"has_embedding": True, "dim": embed_dim} for _ in range(n1)
            ],
            seed=42,
        )
        features2 = features_factory(
            n_detections=n2,
            embed_configs=[
                {"has_embedding": True, "dim": embed_dim} for _ in range(n2)
            ],
            seed=100,
        )

        result = compute_vectorized_embedding_distances(features1, features2)

        # Should complete and return correct shape
        assert result.shape == (n1, n2)

        # All should be valid since we have valid embeddings
        assert np.all(~np.isnan(result))
        assert np.all(result >= 0)
        assert np.all(result <= 1.0)

    def test_memory_efficiency_sparse_valid(self, features_factory):
        """Test memory efficiency with sparse valid embeddings."""
        n1, n2 = 20, 25

        # Most embeddings invalid, only a few valid
        embed_configs1 = [{"has_embedding": i < 3} for i in range(n1)]
        embed_configs2 = [{"has_embedding": i < 4} for i in range(n2)]

        features1 = features_factory(n_detections=n1, embed_configs=embed_configs1)
        features2 = features_factory(n_detections=n2, embed_configs=embed_configs2)

        result = compute_vectorized_embedding_distances(features1, features2)

        assert result.shape == (n1, n2)

        # Only the top-left corner should have valid distances
        assert np.all(~np.isnan(result[:3, :4]))  # Valid region
        assert np.all(np.isnan(result[3:, :]))  # Invalid rows
        assert np.all(np.isnan(result[:, 4:]))  # Invalid columns


class TestComputeVectorizedEmbeddingDistancesIntegration:
    """Test integration with existing codebase."""

    def test_match_original_distance_matrix(self, detection_factory, features_factory):
        """Test that results match original pairwise distance computations."""
        from mouse_tracking.matching.core import Detection

        # Create several detections with various embedding configurations
        detections = [
            detection_factory(pose_idx=0, embed_dim=32, seed=42),  # Valid embedding
            detection_factory(pose_idx=1, embed_dim=32, seed=100),  # Valid embedding
            detection_factory(pose_idx=2, has_embedding=False),  # No embedding
        ]

        # Manually set the third detection to have zero embedding (invalid)
        detections[2].embed = np.zeros(32)

        # Compute original distance matrix
        n = len(detections)
        original_matrix = np.full((n, n), np.nan)

        for i in range(n):
            for j in range(n):
                original_matrix[i, j] = Detection.embed_distance(
                    detections[i].embed, detections[j].embed
                )

        # Compute vectorized distance matrix
        features = features_factory(
            n_detections=n, embed_configs=[{"has_embedding": False}] * n
        )
        features.detections = detections
        features.embeddings = np.array([det.embed for det in detections])

        # Update valid masks based on embeddings
        features.valid_embed_masks = ~np.all(features.embeddings == 0, axis=-1)

        vectorized_matrix = compute_vectorized_embedding_distances(features, features)

        # Should match original matrix (handling NaN values)
        assert original_matrix.shape == vectorized_matrix.shape

        # Check NaN positions match
        orig_nan_mask = np.isnan(original_matrix)
        vect_nan_mask = np.isnan(vectorized_matrix)
        assert np.array_equal(orig_nan_mask, vect_nan_mask)

        # Check non-NaN values match
        valid_mask = ~orig_nan_mask
        if np.any(valid_mask):
            np.testing.assert_allclose(
                original_matrix[valid_mask], vectorized_matrix[valid_mask], rtol=1e-15
            )

    def test_usage_in_compute_vectorized_match_costs(self, features_factory):
        """Test integration with compute_vectorized_match_costs function."""
        from mouse_tracking.matching.vectorized_features import (
            compute_vectorized_match_costs,
        )

        # Create features that would be used in match cost computation
        features1 = features_factory(n_detections=2, seed=42)
        features2 = features_factory(n_detections=3, seed=100)

        # This should not raise any errors and should use our function internally
        result = compute_vectorized_match_costs(features1, features2)

        assert result.shape == (2, 3)
        assert np.all(np.isfinite(result))  # Match costs should be finite

    def test_embedding_dimension_consistency(self, features_factory):
        """Test that embedding dimensions are handled consistently."""
        # Test various embedding dimensions
        dims = [1, 16, 64, 128, 256, 512]

        for dim in dims:
            features1 = features_factory(
                n_detections=2, embed_configs=[{"has_embedding": True, "dim": dim}] * 2
            )
            features2 = features_factory(
                n_detections=2, embed_configs=[{"has_embedding": True, "dim": dim}] * 2
            )

            result = compute_vectorized_embedding_distances(features1, features2)

            assert result.shape == (2, 2)
            assert np.all(~np.isnan(result))
            assert np.all(result >= 0)
            assert np.all(result <= 1.0)
