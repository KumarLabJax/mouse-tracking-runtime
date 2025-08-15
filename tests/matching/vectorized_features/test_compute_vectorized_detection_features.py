"""Tests for VectorizedDetectionFeatures class."""

import numpy as np

from mouse_tracking.matching.vectorized_features import VectorizedDetectionFeatures


class TestVectorizedDetectionFeaturesInit:
    """Test VectorizedDetectionFeatures initialization."""

    def test_init_basic(self, detection_factory):
        """Test basic initialization with valid detections."""
        detections = [
            detection_factory(pose_idx=0, embed_value=0.1),
            detection_factory(pose_idx=1, embed_value=0.2),
            detection_factory(pose_idx=2, embed_value=0.3),
        ]

        features = VectorizedDetectionFeatures(detections)

        assert features.n_detections == 3
        assert features.detections == detections
        assert features.poses.shape == (3, 12, 2)
        assert features.embeddings.shape == (3, 128)
        assert features.valid_pose_masks.shape == (3, 12)
        assert features.valid_embed_masks.shape == (3,)
        assert features._rotated_poses is None
        assert features._seg_images is None

    def test_init_empty_detections(self):
        """Test initialization with empty detection list."""
        features = VectorizedDetectionFeatures([])

        assert features.n_detections == 0
        assert features.detections == []
        assert features.poses.shape == (0,)  # Empty array has shape (0,)
        assert features.embeddings.shape == (0, 0)  # Empty embeddings
        assert (
            features.valid_pose_masks.shape == ()
        )  # Empty array results in scalar shape
        assert features.valid_embed_masks.shape == (0,)

    def test_init_mixed_valid_invalid(self, detection_factory):
        """Test initialization with mixed valid/invalid detections."""
        detections = [
            detection_factory(pose_idx=0, has_pose=True, has_embedding=True),
            detection_factory(pose_idx=1, has_pose=False, has_embedding=True),
            detection_factory(pose_idx=2, has_pose=True, has_embedding=False),
            detection_factory(pose_idx=3, has_pose=False, has_embedding=False),
        ]

        features = VectorizedDetectionFeatures(detections)

        assert features.n_detections == 4
        assert features.poses.shape == (4, 12, 2)
        assert features.embeddings.shape == (4, 128)

        # Check valid masks
        assert features.valid_pose_masks[0].sum() == 12  # All valid
        assert features.valid_pose_masks[1].sum() == 0  # None valid
        assert features.valid_pose_masks[2].sum() == 12  # All valid
        assert features.valid_pose_masks[3].sum() == 0  # None valid

        assert features.valid_embed_masks[0]
        assert features.valid_embed_masks[1]
        assert not features.valid_embed_masks[2]  # No embedding
        assert not features.valid_embed_masks[3]  # No embedding


class TestVectorizedDetectionFeaturesExtractPoses:
    """Test _extract_poses method."""

    def test_extract_poses_valid(self, detection_factory):
        """Test extracting poses with valid data."""
        detections = [
            detection_factory(pose_idx=0, pose_center=(10, 10)),
            detection_factory(pose_idx=1, pose_center=(20, 20)),
        ]

        features = VectorizedDetectionFeatures(detections)

        assert features.poses.shape == (2, 12, 2)
        assert features.poses.dtype == np.float64

        # Check that poses are centered around expected locations
        assert np.abs(features.poses[0].mean(axis=0)[0] - 10) < 10
        assert np.abs(features.poses[0].mean(axis=0)[1] - 10) < 10
        assert np.abs(features.poses[1].mean(axis=0)[0] - 20) < 10
        assert np.abs(features.poses[1].mean(axis=0)[1] - 20) < 10

    def test_extract_poses_none(self, detection_factory):
        """Test extracting poses with None data."""
        detections = [
            detection_factory(pose_idx=0, has_pose=False),
            detection_factory(pose_idx=1, has_pose=False),
        ]

        features = VectorizedDetectionFeatures(detections)

        assert features.poses.shape == (2, 12, 2)
        assert np.all(features.poses == 0)

    def test_extract_poses_mixed(self, detection_factory):
        """Test extracting poses with mixed valid/None data."""
        detections = [
            detection_factory(pose_idx=0, has_pose=True, pose_center=(30, 30)),
            detection_factory(pose_idx=1, has_pose=False),
        ]

        features = VectorizedDetectionFeatures(detections)

        assert features.poses.shape == (2, 12, 2)
        assert not np.all(features.poses[0] == 0)  # First has valid pose
        assert np.all(features.poses[1] == 0)  # Second is zeros


class TestVectorizedDetectionFeaturesExtractEmbeddings:
    """Test _extract_embeddings method."""

    def test_extract_embeddings_valid(self, detection_factory):
        """Test extracting embeddings with valid data."""
        detections = [
            detection_factory(pose_idx=0, embed_dim=64, embed_value=0.1),
            detection_factory(pose_idx=1, embed_dim=64, embed_value=0.2),
        ]

        features = VectorizedDetectionFeatures(detections)

        assert features.embeddings.shape == (2, 64)
        assert features.embeddings.dtype == np.float64
        assert np.allclose(features.embeddings[0], 0.1)
        assert np.allclose(features.embeddings[1], 0.2)

    def test_extract_embeddings_none(self, detection_factory):
        """Test extracting embeddings with None data."""
        detections = [
            detection_factory(pose_idx=0, has_embedding=False),
            detection_factory(pose_idx=1, has_embedding=False),
        ]

        features = VectorizedDetectionFeatures(detections)

        assert features.embeddings.shape == (2, 0)  # Empty embeddings

    def test_extract_embeddings_mixed(self, detection_factory):
        """Test extracting embeddings with mixed valid/None data."""
        detections = [
            detection_factory(
                pose_idx=0, has_embedding=True, embed_dim=32, embed_value=0.5
            ),
            detection_factory(pose_idx=1, has_embedding=False),
        ]

        features = VectorizedDetectionFeatures(detections)

        assert features.embeddings.shape == (2, 32)
        assert np.allclose(features.embeddings[0], 0.5)
        assert np.all(features.embeddings[1] == 0)  # Default zeros

    def test_extract_embeddings_dimension_mismatch(self, mock_detection):
        """Test extracting embeddings with dimension mismatches."""
        det1 = mock_detection(pose_idx=0, embed=np.array([1, 2, 3]))
        det2 = mock_detection(pose_idx=1, embed=np.array([4, 5]))  # Different dimension

        detections = [det1, det2]

        features = VectorizedDetectionFeatures(detections)

        # Should use first valid embedding dimension
        assert features.embeddings.shape == (2, 3)
        assert np.allclose(features.embeddings[0], [1, 2, 3])
        assert np.all(features.embeddings[1] == 0)  # Mismatched dimension becomes zeros


class TestVectorizedDetectionFeaturesComputeValidMasks:
    """Test mask computation methods."""

    def test_compute_valid_pose_masks(self, detection_factory):
        """Test computing valid pose masks."""
        detections = [
            detection_factory(pose_idx=0, has_pose=True),
            detection_factory(pose_idx=1, has_pose=False),
        ]

        features = VectorizedDetectionFeatures(detections)

        assert features.valid_pose_masks.shape == (2, 12)
        assert features.valid_pose_masks.dtype == bool
        assert np.all(features.valid_pose_masks[0])  # All valid
        assert not np.any(features.valid_pose_masks[1])  # None valid

    def test_compute_valid_embed_masks(self, detection_factory):
        """Test computing valid embedding masks."""
        detections = [
            detection_factory(pose_idx=0, has_embedding=True, embed_value=0.5),
            detection_factory(pose_idx=1, has_embedding=False),
        ]

        features = VectorizedDetectionFeatures(detections)

        assert features.valid_embed_masks.shape == (2,)
        assert features.valid_embed_masks.dtype == bool
        assert features.valid_embed_masks[0]
        assert not features.valid_embed_masks[1]

    def test_compute_valid_embed_masks_empty(self, detection_factory):
        """Test computing valid embedding masks with empty embeddings."""
        detections = [
            detection_factory(pose_idx=0, has_embedding=False),
            detection_factory(pose_idx=1, has_embedding=False),
        ]

        features = VectorizedDetectionFeatures(detections)

        assert features.valid_embed_masks.shape == (2,)
        assert not np.any(features.valid_embed_masks)


class TestVectorizedDetectionFeaturesProperties:
    """Test properties and basic functionality."""

    def test_data_types(self, detection_factory):
        """Test that arrays have correct data types."""
        detections = [detection_factory(pose_idx=0)]
        features = VectorizedDetectionFeatures(detections)

        assert features.poses.dtype == np.float64
        assert features.embeddings.dtype == np.float64
        assert features.valid_pose_masks.dtype == bool
        assert features.valid_embed_masks.dtype == bool

    def test_shapes_consistency(self, detection_factory):
        """Test that array shapes are consistent."""
        n_detections = 5
        detections = [detection_factory(pose_idx=i) for i in range(n_detections)]
        features = VectorizedDetectionFeatures(detections)

        assert features.poses.shape[0] == n_detections
        assert features.embeddings.shape[0] == n_detections
        assert features.valid_pose_masks.shape[0] == n_detections
        assert features.valid_embed_masks.shape[0] == n_detections

    def test_caching_initialization(self, detection_factory):
        """Test that cached properties are initialized correctly."""
        detections = [detection_factory(pose_idx=0)]
        features = VectorizedDetectionFeatures(detections)

        assert features._rotated_poses is None
        assert features._seg_images is None

    def test_zero_keypoints_pose(self, mock_detection):
        """Test handling of poses with partial zero keypoints."""
        # Create pose with some zero keypoints
        pose = np.random.random((12, 2)) * 100
        pose[5:8] = 0  # Set some keypoints to zero

        detection = mock_detection(pose_idx=0, pose=pose)
        features = VectorizedDetectionFeatures([detection])

        # Valid mask should be False for zero keypoints
        assert np.all(features.valid_pose_masks[0, :5])  # First 5 are valid
        assert not np.any(features.valid_pose_masks[0, 5:8])  # These are invalid
        assert np.all(features.valid_pose_masks[0, 8:])  # Rest are valid

    def test_zero_embedding_handling(self, mock_detection):
        """Test handling of zero embeddings."""
        # Create embedding with some zeros
        embed = np.array([0.1, 0.2, 0.0, 0.0, 0.3])

        detection = mock_detection(pose_idx=0, embed=embed)
        features = VectorizedDetectionFeatures([detection])

        # Should still be considered valid (only all-zeros are invalid)
        assert features.valid_embed_masks[0]

        # But all-zeros should be invalid
        detection_zeros = mock_detection(pose_idx=0, embed=np.zeros(5))
        features_zeros = VectorizedDetectionFeatures([detection_zeros])
        assert not features_zeros.valid_embed_masks[0]


class TestVectorizedDetectionFeaturesEdgeCases:
    """Test edge cases and error conditions."""

    def test_single_detection(self, detection_factory):
        """Test with single detection."""
        detections = [detection_factory(pose_idx=0)]
        features = VectorizedDetectionFeatures(detections)

        assert features.n_detections == 1
        assert features.poses.shape == (1, 12, 2)
        assert features.embeddings.shape == (1, 128)
        assert features.valid_pose_masks.shape == (1, 12)
        assert features.valid_embed_masks.shape == (1,)

    def test_large_number_detections(self, detection_factory):
        """Test with many detections."""
        n_detections = 100
        detections = [detection_factory(pose_idx=i) for i in range(n_detections)]
        features = VectorizedDetectionFeatures(detections)

        assert features.n_detections == n_detections
        assert features.poses.shape == (n_detections, 12, 2)
        assert features.embeddings.shape == (n_detections, 128)

    def test_all_invalid_data(self, detection_factory):
        """Test with all invalid data."""
        detections = [
            detection_factory(pose_idx=i, has_pose=False, has_embedding=False)
            for i in range(3)
        ]
        features = VectorizedDetectionFeatures(detections)

        assert features.n_detections == 3
        assert np.all(features.poses == 0)
        assert features.embeddings.shape == (3, 0)  # Empty embeddings
        assert not np.any(features.valid_pose_masks)
        assert not np.any(features.valid_embed_masks)

    def test_different_embedding_dimensions(self, mock_detection):
        """Test behavior with different embedding dimensions."""
        # First detection has embedding
        det1 = mock_detection(pose_idx=0, embed=np.array([1, 2, 3, 4]))

        # Second detection has different dimension (should become zeros)
        det2 = mock_detection(pose_idx=1, embed=np.array([5, 6]))

        # Third detection has no embedding
        det3 = mock_detection(pose_idx=2, embed=None)

        detections = [det1, det2, det3]
        features = VectorizedDetectionFeatures(detections)

        # Should use first valid embedding dimension
        assert features.embeddings.shape == (3, 4)
        assert np.allclose(features.embeddings[0], [1, 2, 3, 4])
        assert np.all(features.embeddings[1] == 0)  # Mismatched dimension
        assert np.all(features.embeddings[2] == 0)  # None embedding

        # Valid masks should reflect this
        assert features.valid_embed_masks[0]
        assert not features.valid_embed_masks[1]
        assert not features.valid_embed_masks[2]
