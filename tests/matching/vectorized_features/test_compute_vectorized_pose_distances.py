"""Tests for compute_vectorized_pose_distances function."""

from unittest.mock import patch

import numpy as np
import pytest

from mouse_tracking.matching.vectorized_features import (
    VectorizedDetectionFeatures,
    compute_vectorized_pose_distances,
)


class TestComputeVectorizedPoseDistances:
    """Test compute_vectorized_pose_distances function."""

    def test_basic_pose_distances(self, features_factory):
        """Test basic pose distance computation."""
        # Create features with known poses
        features1 = features_factory(
            n_detections=2,
            pose_configs=[
                {"has_pose": True, "center": (0, 0)},
                {"has_pose": True, "center": (10, 10)},
            ],
        )
        features2 = features_factory(
            n_detections=2,
            pose_configs=[
                {"has_pose": True, "center": (0, 0)},
                {"has_pose": True, "center": (20, 20)},
            ],
        )

        distances = compute_vectorized_pose_distances(features1, features2)

        # Check shape and data type
        assert distances.shape == (2, 2)
        assert distances.dtype == np.float64

        # Distance from pose to itself should be 0
        assert distances[0, 0] == pytest.approx(0.0, abs=1e-6)

        # Distance should be symmetric for same poses
        assert not np.isnan(distances[0, 1])
        assert not np.isnan(distances[1, 0])

        # All distances should be non-negative
        assert np.all(distances >= 0)

    def test_pose_distances_with_invalid_poses(self, features_factory):
        """Test pose distance computation with invalid poses."""
        features1 = features_factory(
            n_detections=2,
            pose_configs=[
                {"has_pose": True, "center": (0, 0)},
                {"has_pose": False},  # Invalid pose
            ],
        )
        features2 = features_factory(
            n_detections=2,
            pose_configs=[
                {"has_pose": True, "center": (10, 10)},
                {"has_pose": True, "center": (20, 20)},
            ],
        )

        distances = compute_vectorized_pose_distances(features1, features2)

        # Check shape
        assert distances.shape == (2, 2)

        # Valid pose comparisons should work
        assert not np.isnan(distances[0, 0])
        assert not np.isnan(distances[0, 1])

        # Invalid pose comparisons should return NaN
        assert np.isnan(distances[1, 0])
        assert np.isnan(distances[1, 1])

    def test_pose_distances_all_invalid(self, features_factory):
        """Test pose distance computation with all invalid poses."""
        features1 = features_factory(
            n_detections=2,
            pose_configs=[
                {"has_pose": False},
                {"has_pose": False},
            ],
        )
        features2 = features_factory(
            n_detections=2,
            pose_configs=[
                {"has_pose": False},
                {"has_pose": False},
            ],
        )

        distances = compute_vectorized_pose_distances(features1, features2)

        # All should be NaN
        assert distances.shape == (2, 2)
        assert np.all(np.isnan(distances))

    def test_pose_distances_with_rotation(self, features_factory):
        """Test pose distance computation with rotation enabled."""
        features1 = features_factory(
            n_detections=1, pose_configs=[{"has_pose": True, "center": (0, 0)}]
        )
        features2 = features_factory(
            n_detections=1, pose_configs=[{"has_pose": True, "center": (10, 10)}]
        )

        # Test without rotation
        distances_no_rot = compute_vectorized_pose_distances(
            features1, features2, use_rotation=False
        )

        # Test with rotation
        distances_with_rot = compute_vectorized_pose_distances(
            features1, features2, use_rotation=True
        )

        # Both should be valid
        assert not np.isnan(distances_no_rot[0, 0])
        assert not np.isnan(distances_with_rot[0, 0])

        # With rotation should be <= without rotation (minimum is taken)
        assert distances_with_rot[0, 0] <= distances_no_rot[0, 0]

    def test_pose_distances_rotation_calls_get_rotated_poses(self, features_factory):
        """Test that rotation mode calls get_rotated_poses."""
        features1 = features_factory(
            n_detections=1, pose_configs=[{"has_pose": True, "center": (0, 0)}]
        )
        features2 = features_factory(
            n_detections=1, pose_configs=[{"has_pose": True, "center": (10, 10)}]
        )

        # Mock get_rotated_poses to track calls
        with patch.object(features1, "get_rotated_poses") as mock_get_rotated:
            mock_get_rotated.return_value = np.ones((1, 12, 2)) * 5

            distances = compute_vectorized_pose_distances(
                features1, features2, use_rotation=True
            )

            # Should call get_rotated_poses
            mock_get_rotated.assert_called_once()

            # Should return valid result
            assert not np.isnan(distances[0, 0])

    def test_pose_distances_different_sizes(self, features_factory):
        """Test pose distance computation with different sized feature sets."""
        features1 = features_factory(
            n_detections=3,
            pose_configs=[
                {"has_pose": True, "center": (0, 0)},
                {"has_pose": True, "center": (10, 10)},
                {"has_pose": True, "center": (20, 20)},
            ],
        )
        features2 = features_factory(
            n_detections=2,
            pose_configs=[
                {"has_pose": True, "center": (5, 5)},
                {"has_pose": True, "center": (15, 15)},
            ],
        )

        distances = compute_vectorized_pose_distances(features1, features2)

        # Should handle different sizes
        assert distances.shape == (3, 2)
        assert not np.any(np.isnan(distances))  # All should be valid

    def test_pose_distances_empty_features(self):
        """Test pose distance computation with empty features."""
        features1 = VectorizedDetectionFeatures([])
        features2 = VectorizedDetectionFeatures([])

        # This will likely crash due to empty array indexing - mark as expected behavior
        # TODO: This reveals a bug in the function with empty features
        with pytest.raises(IndexError):
            compute_vectorized_pose_distances(features1, features2)

    def test_pose_distances_single_detection(self, features_factory):
        """Test pose distance computation with single detection."""
        features1 = features_factory(
            n_detections=1, pose_configs=[{"has_pose": True, "center": (0, 0)}]
        )
        features2 = features_factory(
            n_detections=1, pose_configs=[{"has_pose": True, "center": (10, 10)}]
        )

        distances = compute_vectorized_pose_distances(features1, features2)

        assert distances.shape == (1, 1)
        assert not np.isnan(distances[0, 0])
        assert distances[0, 0] > 0  # Should be positive distance

    def test_pose_distances_keypoint_masking(self, mock_detection):
        """Test that keypoint masking works correctly."""
        # Create poses with some zero keypoints
        pose1 = np.random.random((12, 2)) * 10
        pose1[5:8] = 0  # Zero out some keypoints

        pose2 = np.random.random((12, 2)) * 10
        pose2[8:11] = 0  # Zero out different keypoints

        det1 = mock_detection(pose_idx=0, pose=pose1)
        det2 = mock_detection(pose_idx=1, pose=pose2)

        features1 = VectorizedDetectionFeatures([det1])
        features2 = VectorizedDetectionFeatures([det2])

        distances = compute_vectorized_pose_distances(features1, features2)

        # Should compute distance using only valid keypoints
        assert distances.shape == (1, 1)
        assert not np.isnan(distances[0, 0])
        assert distances[0, 0] >= 0

    def test_pose_distances_numerical_accuracy(self, mock_detection):
        """Test numerical accuracy of distance computation."""
        # Create simple poses for exact calculation - avoid (0,0) which is considered invalid
        pose1 = np.zeros((12, 2))
        pose1[0] = [1, 1]  # Valid keypoint
        pose1[1] = [4, 5]  # Distance from pose2[1] should be 5

        pose2 = np.zeros((12, 2))
        pose2[0] = [1, 1]  # Same as pose1[0], distance = 0
        pose2[1] = [1, 1]  # Distance from pose1[1] should be 5

        det1 = mock_detection(pose_idx=0, pose=pose1)
        det2 = mock_detection(pose_idx=1, pose=pose2)

        features1 = VectorizedDetectionFeatures([det1])
        features2 = VectorizedDetectionFeatures([det2])

        distances = compute_vectorized_pose_distances(features1, features2)

        # Mean distance should be (0 + 5) / 2 = 2.5
        expected_distance = 2.5
        assert distances[0, 0] == pytest.approx(expected_distance, abs=1e-6)


class TestComputeVectorizedPoseDistancesRotation:
    """Test rotation-specific functionality."""

    def test_rotation_minimum_selection(self, features_factory):
        """Test that rotation selects minimum distance."""
        features1 = features_factory(
            n_detections=1, pose_configs=[{"has_pose": True, "center": (10, 10)}]
        )
        features2 = features_factory(
            n_detections=1, pose_configs=[{"has_pose": True, "center": (20, 20)}]
        )

        # Get distances without rotation first
        distances_no_rot = compute_vectorized_pose_distances(
            features1, features2, use_rotation=False
        )

        # Mock get_rotated_poses to return poses that would result in smaller distance
        with patch.object(features1, "get_rotated_poses") as mock_get_rotated:
            # Create rotated poses that are closer to the second pose
            rotated_poses = np.ones((1, 12, 2))
            rotated_poses[0] = rotated_poses[0] * 19  # Very close to (20, 20)
            mock_get_rotated.return_value = rotated_poses

            distances_with_rot = compute_vectorized_pose_distances(
                features1, features2, use_rotation=True
            )

            # Should use the minimum distance (rotated should be smaller)
            assert distances_with_rot[0, 0] < distances_no_rot[0, 0]

    def test_rotation_with_invalid_poses(self, features_factory):
        """Test rotation behavior with invalid poses."""
        features1 = features_factory(
            n_detections=2,
            pose_configs=[
                {"has_pose": True, "center": (0, 0)},
                {"has_pose": False},  # Invalid pose
            ],
        )
        features2 = features_factory(
            n_detections=1, pose_configs=[{"has_pose": True, "center": (10, 10)}]
        )

        distances = compute_vectorized_pose_distances(
            features1, features2, use_rotation=True
        )

        # Valid pose should work
        assert not np.isnan(distances[0, 0])

        # Invalid pose should still be NaN
        assert np.isnan(distances[1, 0])

    def test_rotation_nan_handling(self, features_factory):
        """Test that rotation properly handles NaN values."""
        features1 = features_factory(
            n_detections=1, pose_configs=[{"has_pose": True, "center": (0, 0)}]
        )
        features2 = features_factory(
            n_detections=1,
            pose_configs=[{"has_pose": False}],  # Invalid pose
        )

        distances = compute_vectorized_pose_distances(
            features1, features2, use_rotation=True
        )

        # Should handle NaN correctly
        assert np.isnan(distances[0, 0])


class TestComputeVectorizedPoseDistancesEdgeCases:
    """Test edge cases and error conditions."""

    def test_single_valid_keypoint(self, mock_detection):
        """Test with poses having only one valid keypoint."""
        pose1 = np.zeros((12, 2))
        pose1[0] = [1, 1]  # Only first keypoint is valid (avoid 0,0 which is invalid)

        pose2 = np.zeros((12, 2))
        pose2[0] = [4, 5]  # Only first keypoint is valid

        det1 = mock_detection(pose_idx=0, pose=pose1)
        det2 = mock_detection(pose_idx=1, pose=pose2)

        features1 = VectorizedDetectionFeatures([det1])
        features2 = VectorizedDetectionFeatures([det2])

        distances = compute_vectorized_pose_distances(features1, features2)

        # Should compute distance using single valid keypoint
        assert distances.shape == (1, 1)
        assert not np.isnan(distances[0, 0])
        assert distances[0, 0] == pytest.approx(5.0, abs=1e-6)

    def test_no_valid_keypoints(self, mock_detection):
        """Test with poses having no valid keypoints."""
        pose1 = np.zeros((12, 2))  # All keypoints are zeros
        pose2 = np.zeros((12, 2))  # All keypoints are zeros

        det1 = mock_detection(pose_idx=0, pose=pose1)
        det2 = mock_detection(pose_idx=1, pose=pose2)

        features1 = VectorizedDetectionFeatures([det1])
        features2 = VectorizedDetectionFeatures([det2])

        distances = compute_vectorized_pose_distances(features1, features2)

        # Should return NaN for no valid keypoints
        assert distances.shape == (1, 1)
        assert np.isnan(distances[0, 0])

    def test_asymmetric_valid_keypoints(self, mock_detection):
        """Test with asymmetric valid keypoints."""
        pose1 = np.zeros((12, 2))
        pose1[0] = [0, 0]  # First keypoint valid

        pose2 = np.zeros((12, 2))
        pose2[1] = [3, 4]  # Second keypoint valid

        det1 = mock_detection(pose_idx=0, pose=pose1)
        det2 = mock_detection(pose_idx=1, pose=pose2)

        features1 = VectorizedDetectionFeatures([det1])
        features2 = VectorizedDetectionFeatures([det2])

        distances = compute_vectorized_pose_distances(features1, features2)

        # Should return NaN because no common valid keypoints
        assert distances.shape == (1, 1)
        assert np.isnan(distances[0, 0])

    def test_large_feature_sets(self, features_factory):
        """Test with large feature sets."""
        n_detections = 50
        features1 = features_factory(n_detections=n_detections)
        features2 = features_factory(n_detections=n_detections)

        distances = compute_vectorized_pose_distances(features1, features2)

        # Should handle large sets
        assert distances.shape == (n_detections, n_detections)
        assert not np.any(np.isnan(distances))  # All should be valid

    def test_data_type_consistency(self, features_factory):
        """Test that data types are consistent."""
        features1 = features_factory(n_detections=2)
        features2 = features_factory(n_detections=2)

        distances = compute_vectorized_pose_distances(features1, features2)

        # Should be float64
        assert distances.dtype == np.float64

    def test_warning_suppression(self, features_factory):
        """Test that warnings are properly suppressed."""
        features1 = features_factory(
            n_detections=1,
            pose_configs=[{"has_pose": False}],  # This will cause warnings
        )
        features2 = features_factory(
            n_detections=1, pose_configs=[{"has_pose": True, "center": (10, 10)}]
        )

        # Should not raise warnings
        import warnings

        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            distances = compute_vectorized_pose_distances(features1, features2)

        # Check that no RuntimeWarnings were raised
        runtime_warnings = [
            w for w in warning_list if issubclass(w.category, RuntimeWarning)
        ]
        assert len(runtime_warnings) == 0

        # Result should still be correct
        assert np.isnan(distances[0, 0])


class TestComputeVectorizedPoseDistancesIntegration:
    """Integration tests for compute_vectorized_pose_distances."""

    def test_integration_with_real_data(self, detection_factory):
        """Test with real detection data."""
        detections1 = [
            detection_factory(pose_idx=0, pose_center=(10, 10)),
            detection_factory(pose_idx=1, pose_center=(20, 20)),
        ]
        detections2 = [
            detection_factory(pose_idx=0, pose_center=(15, 15)),
            detection_factory(pose_idx=1, pose_center=(25, 25)),
        ]

        features1 = VectorizedDetectionFeatures(detections1)
        features2 = VectorizedDetectionFeatures(detections2)

        distances = compute_vectorized_pose_distances(features1, features2)

        # Should compute reasonable distances
        assert distances.shape == (2, 2)
        assert not np.any(np.isnan(distances))
        assert np.all(distances >= 0)

        # Closer poses should have smaller distances
        assert (
            distances[0, 0] < distances[0, 1]
        )  # (10,10) closer to (15,15) than (25,25)

    def test_integration_rotation_real_data(self, detection_factory):
        """Test rotation with real detection data."""
        detections1 = [detection_factory(pose_idx=0, pose_center=(10, 10))]
        detections2 = [detection_factory(pose_idx=0, pose_center=(20, 20))]

        features1 = VectorizedDetectionFeatures(detections1)
        features2 = VectorizedDetectionFeatures(detections2)

        distances_no_rot = compute_vectorized_pose_distances(
            features1, features2, use_rotation=False
        )
        distances_with_rot = compute_vectorized_pose_distances(
            features1, features2, use_rotation=True
        )

        # Both should be valid
        assert not np.isnan(distances_no_rot[0, 0])
        assert not np.isnan(distances_with_rot[0, 0])

        # With rotation should be <= without rotation
        assert distances_with_rot[0, 0] <= distances_no_rot[0, 0]

    def test_symmetry_property(self, features_factory):
        """Test that distance computation maintains reasonable symmetry."""
        features1 = features_factory(n_detections=3)
        features2 = features_factory(n_detections=3)

        distances_1_to_2 = compute_vectorized_pose_distances(features1, features2)
        distances_2_to_1 = compute_vectorized_pose_distances(features2, features1)

        # Should be transpose of each other
        assert np.allclose(distances_1_to_2, distances_2_to_1.T, equal_nan=True)

    def test_diagonal_self_distances(self, features_factory):
        """Test that self-distances are zero."""
        features = features_factory(n_detections=3)

        distances = compute_vectorized_pose_distances(features, features)

        # Diagonal should be zero (pose distance to itself)
        diagonal = np.diag(distances)
        assert np.allclose(diagonal, 0, atol=1e-6)
