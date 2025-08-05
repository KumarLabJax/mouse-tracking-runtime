"""Tests for VectorizedDetectionFeatures.get_rotated_poses method."""

from unittest.mock import patch

import numpy as np

from mouse_tracking.matching.vectorized_features import VectorizedDetectionFeatures


class TestGetRotatedPoses:
    """Test get_rotated_poses method."""

    def test_get_rotated_poses_basic(self, detection_factory):
        """Test basic rotation functionality."""
        detections = [
            detection_factory(pose_idx=0, pose_center=(50, 50)),
            detection_factory(pose_idx=1, pose_center=(100, 100)),
        ]

        features = VectorizedDetectionFeatures(detections)

        # Mock the Detection.rotate_pose method
        with patch("mouse_tracking.matching.core.Detection.rotate_pose") as mock_rotate:
            # Set up mock return values (12 keypoints, 2 coordinates)
            mock_rotate.side_effect = [
                np.ones((12, 2)) * 1,  # Mock rotated pose for first detection
                np.ones((12, 2)) * 2,  # Mock rotated pose for second detection
            ]

            rotated_poses = features.get_rotated_poses()

            # Check that Detection.rotate_pose was called correctly
            assert mock_rotate.call_count == 2

            # Check the calls were made with correct parameters
            calls = mock_rotate.call_args_list
            assert calls[0][0][1] == 180  # Second argument should be 180 degrees
            assert calls[1][0][1] == 180  # Second argument should be 180 degrees

            # Check the returned shape
            assert rotated_poses.shape == (2, 12, 2)
            assert rotated_poses.dtype == np.float64

            # Check that the cached result is stored
            assert features._rotated_poses is rotated_poses

    def test_get_rotated_poses_caching(self, detection_factory):
        """Test that rotated poses are cached."""
        detections = [detection_factory(pose_idx=0, pose_center=(50, 50))]
        features = VectorizedDetectionFeatures(detections)

        with patch("mouse_tracking.matching.core.Detection.rotate_pose") as mock_rotate:
            mock_rotate.return_value = np.ones((12, 2)) * 5  # Correct shape

            # First call should compute
            rotated_poses1 = features.get_rotated_poses()
            assert mock_rotate.call_count == 1

            # Second call should use cache
            rotated_poses2 = features.get_rotated_poses()
            assert mock_rotate.call_count == 1  # Should not be called again

            # Should return the same object
            assert rotated_poses1 is rotated_poses2

    def test_get_rotated_poses_none_poses(self, detection_factory):
        """Test handling of None poses."""
        detections = [
            detection_factory(pose_idx=0, has_pose=True, pose_center=(50, 50)),
            detection_factory(pose_idx=1, has_pose=False),  # No pose
        ]

        features = VectorizedDetectionFeatures(detections)

        with patch("mouse_tracking.matching.core.Detection.rotate_pose") as mock_rotate:
            mock_rotate.return_value = np.ones((12, 2)) * 7  # Correct shape

            rotated_poses = features.get_rotated_poses()

            # Should only call rotate_pose for the detection with a pose
            assert mock_rotate.call_count == 1

            # Check the shape
            assert rotated_poses.shape == (2, 12, 2)

            # Second detection should have zeros (unchanged from original)
            assert np.all(rotated_poses[1] == 0)

    def test_get_rotated_poses_all_none(self, detection_factory):
        """Test handling when all poses are None."""
        detections = [
            detection_factory(pose_idx=0, has_pose=False),
            detection_factory(pose_idx=1, has_pose=False),
        ]

        features = VectorizedDetectionFeatures(detections)

        with patch("mouse_tracking.matching.core.Detection.rotate_pose") as mock_rotate:
            rotated_poses = features.get_rotated_poses()

            # Should not call rotate_pose at all
            assert mock_rotate.call_count == 0

            # All poses should be zeros
            assert np.all(rotated_poses == 0)
            assert rotated_poses.shape == (2, 12, 2)

    def test_get_rotated_poses_empty_detections(self):
        """Test handling of empty detections list."""
        features = VectorizedDetectionFeatures([])

        with patch("mouse_tracking.matching.core.Detection.rotate_pose") as mock_rotate:
            rotated_poses = features.get_rotated_poses()

            # Should not call rotate_pose
            assert mock_rotate.call_count == 0

            # Should return empty array matching poses shape
            assert rotated_poses.shape == (0,)
            assert np.array_equal(rotated_poses, features.poses)

    def test_get_rotated_poses_uses_detection_rotate_pose(self, detection_factory):
        """Test that the method uses Detection.rotate_pose correctly."""
        detections = [detection_factory(pose_idx=0, pose_center=(30, 40))]
        features = VectorizedDetectionFeatures(detections)

        with patch("mouse_tracking.matching.core.Detection.rotate_pose") as mock_rotate:
            mock_rotate.return_value = np.ones((12, 2)) * 5  # Mock return value

            rotated_poses = features.get_rotated_poses()

            # Check that rotate_pose was called with correct arguments
            assert mock_rotate.call_count == 1
            call_args = mock_rotate.call_args

            # First argument should be the pose
            pose_arg = call_args[0][0]
            assert pose_arg.shape == (12, 2)

            # Second argument should be 180 degrees
            assert call_args[0][1] == 180

            # Result should use the mocked return value
            assert np.allclose(rotated_poses[0], 5)

    def test_get_rotated_poses_mixed_valid_invalid(self, detection_factory):
        """Test with mixed valid and invalid poses."""
        detections = [
            detection_factory(pose_idx=0, has_pose=True, pose_center=(10, 20)),
            detection_factory(pose_idx=1, has_pose=False),
            detection_factory(pose_idx=2, has_pose=True, pose_center=(30, 40)),
            detection_factory(pose_idx=3, has_pose=False),
        ]

        features = VectorizedDetectionFeatures(detections)

        with patch("mouse_tracking.matching.core.Detection.rotate_pose") as mock_rotate:
            mock_rotate.side_effect = [
                np.ones((12, 2)) * 1,  # For detection 0
                np.ones((12, 2)) * 2,  # For detection 2
            ]

            rotated_poses = features.get_rotated_poses()

            # Should call rotate_pose twice (for detections 0 and 2)
            assert mock_rotate.call_count == 2

            # Check the results
            assert rotated_poses.shape == (4, 12, 2)
            assert np.allclose(rotated_poses[0], 1)  # First detection
            assert np.all(rotated_poses[1] == 0)  # Second detection (None)
            assert np.allclose(rotated_poses[2], 2)  # Third detection
            assert np.all(rotated_poses[3] == 0)  # Fourth detection (None)

    def test_get_rotated_poses_circular_import_handling(self, detection_factory):
        """Test that circular import is handled correctly."""
        detections = [detection_factory(pose_idx=0, pose_center=(50, 50))]
        features = VectorizedDetectionFeatures(detections)

        # This test mainly verifies that the import is deferred and doesn't cause issues
        # The actual import happens inside the method
        with patch("mouse_tracking.matching.core.Detection.rotate_pose") as mock_rotate:
            mock_rotate.return_value = np.zeros((12, 2))

            rotated_poses = features.get_rotated_poses()

            # Should successfully call the method
            assert mock_rotate.call_count == 1
            assert rotated_poses.shape == (1, 12, 2)

    def test_get_rotated_poses_preserves_original_poses(self, detection_factory):
        """Test that original poses are not modified."""
        detections = [detection_factory(pose_idx=0, pose_center=(50, 50))]
        features = VectorizedDetectionFeatures(detections)

        # Store original poses
        original_poses = features.poses.copy()

        with patch("mouse_tracking.matching.core.Detection.rotate_pose") as mock_rotate:
            mock_rotate.return_value = (
                np.ones((12, 2)) * 100
            )  # Very different from original

            rotated_poses = features.get_rotated_poses()

            # Original poses should be unchanged
            assert np.array_equal(features.poses, original_poses)

            # Rotated poses should be different
            assert not np.array_equal(rotated_poses, original_poses)


class TestGetRotatedPosesIntegration:
    """Integration tests for get_rotated_poses method."""

    def test_get_rotated_poses_real_rotation(self, detection_factory):
        """Test with real rotation (no mocking)."""
        # Create a simple test pose
        pose = np.array(
            [
                [0, 0],  # Point at origin
                [10, 0],  # Point to the right
                [0, 10],  # Point up
                [10, 10],  # Point diagonal
            ]
            + [[0, 0]] * 8
        )  # Fill remaining keypoints with zeros

        # Create detection with this pose
        detection = detection_factory(pose_idx=0, has_pose=True)
        detection.pose = pose

        features = VectorizedDetectionFeatures([detection])

        # Get rotated poses (this will use the actual rotate_pose method)
        rotated_poses = features.get_rotated_poses()

        # Check that we got a result
        assert rotated_poses.shape == (1, 12, 2)

        # The rotation should have been applied
        # (We don't test the exact rotation math here since that's tested in Detection.rotate_pose)
        assert not np.array_equal(rotated_poses[0], pose)

    def test_get_rotated_poses_consistency(self, detection_factory):
        """Test that method produces consistent results."""
        detections = [
            detection_factory(pose_idx=0, pose_center=(25, 25)),
            detection_factory(pose_idx=1, pose_center=(75, 75)),
        ]

        features = VectorizedDetectionFeatures(detections)

        # Get rotated poses multiple times
        rotated_poses1 = features.get_rotated_poses()
        rotated_poses2 = features.get_rotated_poses()
        rotated_poses3 = features.get_rotated_poses()

        # All should be identical (due to caching)
        assert np.array_equal(rotated_poses1, rotated_poses2)
        assert np.array_equal(rotated_poses2, rotated_poses3)
        assert rotated_poses1 is rotated_poses2  # Same object due to caching

    def test_get_rotated_poses_data_types(self, detection_factory):
        """Test that data types are preserved correctly."""
        detections = [detection_factory(pose_idx=0, pose_center=(50, 50))]
        features = VectorizedDetectionFeatures(detections)

        rotated_poses = features.get_rotated_poses()

        # Should have same data type as original poses
        assert rotated_poses.dtype == features.poses.dtype
        assert rotated_poses.dtype == np.float64
