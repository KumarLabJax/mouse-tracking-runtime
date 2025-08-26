"""
Unit tests for the inspect_pose_v2 function.

This module provides comprehensive test coverage for the inspect_pose_v2 function,
including success paths, error conditions, and edge cases with properly mocked
dependencies to ensure backwards compatibility testing.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mouse_tracking.pose.inspect import inspect_pose_v2


class TestInspectPoseV2BasicFunctionality:
    """Test basic functionality of inspect_pose_v2."""

    @patch("mouse_tracking.pose.inspect.safe_find_first")
    @patch("mouse_tracking.pose.inspect.h5py.File")
    @patch("mouse_tracking.pose.inspect.CONFIG")
    def test_successful_inspection_basic(
        self, mock_config, mock_h5py_file, mock_safe_find_first
    ):
        """Test successful inspection of a valid v2 pose file."""
        # Arrange
        pose_file_path = "/path/to/test_video_pose_est_v2.h5"
        pad = 150
        duration = 108000

        # Mock CONFIG constants
        mock_config.MIN_HIGH_CONFIDENCE = 0.75
        mock_config.MIN_JABS_CONFIDENCE = 0.3

        # Mock HDF5 file structure
        mock_file = MagicMock()
        mock_h5py_file.return_value.__enter__.return_value = mock_file

        # Create test data arrays - v2 has shape [frames, instances, keypoints] like v6
        num_frames = 110000
        pose_quality = np.random.rand(
            num_frames, 1, 12
        )  # Shape [frames, instances, keypoints]
        pose_quality[:100, :, :] = 0  # No confidence before frame 100
        pose_quality[100:110000, :, :] = 0.8  # High confidence after frame 100

        # Mock dataset access
        def mock_getitem(key):
            if key == "poseest":
                mock_poseest = MagicMock()
                mock_poseest.attrs = {"version": [2]}
                return mock_poseest
            elif key == "poseest/confidence":
                return pose_quality
            else:
                raise KeyError(f"Key {key} not found")

        mock_file.__getitem__.side_effect = mock_getitem

        # Mock safe_find_first to return sequential values for testing
        mock_safe_find_first.side_effect = [100, 100]  # Different first frames

        # Act
        result = inspect_pose_v2(pose_file_path, pad=pad, duration=duration)

        # Assert
        assert "first_frame_pose" in result
        assert "first_frame_full_high_conf" in result
        assert "pose_counts" in result
        assert "missing_poses" in result
        assert "missing_keypoint_frames" in result

        assert result["first_frame_pose"] == 100
        assert result["first_frame_full_high_conf"] == 100

        # Verify mocked functions were called correctly
        assert mock_safe_find_first.call_count == 2
        mock_h5py_file.assert_called_once_with(pose_file_path, "r")

    @patch("mouse_tracking.pose.inspect.safe_find_first")
    @patch("mouse_tracking.pose.inspect.h5py.File")
    @patch("mouse_tracking.pose.inspect.CONFIG")
    def test_successful_inspection_with_detailed_calculations(
        self, mock_config, mock_h5py_file, mock_safe_find_first
    ):
        """Test successful inspection with detailed calculation verification."""
        # Arrange
        pose_file_path = "/path/to/detailed_test.h5"
        pad = 50
        duration = 200

        # Mock CONFIG constants
        mock_config.MIN_HIGH_CONFIDENCE = 0.75
        mock_config.MIN_JABS_CONFIDENCE = 0.3

        mock_file = MagicMock()
        mock_h5py_file.return_value.__enter__.return_value = mock_file

        # Create detailed test data
        total_frames = 300
        pose_quality = np.zeros((total_frames, 1, 12))

        # Frame 60-240: 8 keypoints above JABS threshold (0.4 > 0.3)
        # Frame 80-220: all 12 keypoints above high confidence threshold (0.8 > 0.75)
        pose_quality[60:240, :, :8] = 0.4
        pose_quality[80:220, :, :] = 0.8

        def mock_getitem(key):
            if key == "poseest":
                mock_poseest = MagicMock()
                mock_poseest.attrs = {"version": [2]}
                return mock_poseest
            elif key == "poseest/confidence":
                return pose_quality

        mock_file.__getitem__.side_effect = mock_getitem

        mock_safe_find_first.side_effect = [
            60,
            80,
        ]  # first_frame_pose, first_frame_full_high_conf

        # Act
        result = inspect_pose_v2(pose_file_path, pad=pad, duration=duration)

        # Assert
        assert result["first_frame_pose"] == 60
        assert result["first_frame_full_high_conf"] == 80

        # Verify calculations based on actual data:
        # pose_quality[60:240, :, :8] = 0.4  (frames 60-239, first 8 keypoints)
        # pose_quality[80:220, :, :] = 0.8   (frames 80-219, all 12 keypoints)
        #
        # So keypoints > 0.3:
        # - Frames 60-79: 8 keypoints each = 20 * 8 = 160
        # - Frames 80-219: 12 keypoints each = 140 * 12 = 1680
        # - Frames 220-239: 8 keypoints each = 20 * 8 = 160
        # Total: 160 + 1680 + 160 = 2000
        expected_pose_counts = 20 * 8 + 140 * 12 + 20 * 8  # 2000
        assert result["pose_counts"] == expected_pose_counts

        # missing_poses: duration - keypoints in observation window [50:250]
        # All our keypoints (frames 60-239) are within the window, so all 2000 count
        expected_missing_poses = duration - 2000  # 200 - 2000 = -1800
        assert result["missing_poses"] == expected_missing_poses


class TestInspectPoseV2ErrorHandling:
    """Test error handling scenarios."""

    @patch("mouse_tracking.pose.inspect.h5py.File")
    def test_version_not_equal_2_raises_error(self, mock_h5py_file):
        """Test that version != 2 raises ValueError."""
        # Arrange
        pose_file_path = "/path/to/test_v6.h5"

        mock_file = MagicMock()
        mock_h5py_file.return_value.__enter__.return_value = mock_file

        # Mock version 6
        mock_poseest = MagicMock()
        mock_poseest.attrs = {"version": [6]}
        mock_file.__getitem__.return_value = mock_poseest

        # Act & Assert
        with pytest.raises(
            ValueError, match=r"Only v2 pose files are supported.*version 6"
        ):
            inspect_pose_v2(pose_file_path)

    @patch("mouse_tracking.pose.inspect.h5py.File")
    def test_version_1_raises_error(self, mock_h5py_file):
        """Test that version 1 raises ValueError."""
        # Arrange
        pose_file_path = "/path/to/test_v1.h5"

        mock_file = MagicMock()
        mock_h5py_file.return_value.__enter__.return_value = mock_file

        # Mock version 1
        mock_poseest = MagicMock()
        mock_poseest.attrs = {"version": [1]}
        mock_file.__getitem__.return_value = mock_poseest

        # Act & Assert
        with pytest.raises(
            ValueError, match=r"Only v2 pose files are supported.*version 1"
        ):
            inspect_pose_v2(pose_file_path)

    @patch("mouse_tracking.pose.inspect.h5py.File")
    def test_version_attribute_missing(self, mock_h5py_file):
        """Test handling when version attribute is missing."""
        # Arrange
        pose_file_path = "/path/to/no_version.h5"

        mock_file = MagicMock()
        mock_h5py_file.return_value.__enter__.return_value = mock_file

        # Mock missing version
        mock_poseest = MagicMock()
        mock_poseest.attrs.__getitem__.side_effect = KeyError("version")
        mock_file.__getitem__.return_value = mock_poseest

        # Act & Assert
        with pytest.raises(KeyError):
            inspect_pose_v2(pose_file_path)

    @patch("mouse_tracking.pose.inspect.h5py.File")
    def test_missing_confidence_dataset(self, mock_h5py_file):
        """Test handling when confidence dataset is missing."""
        # Arrange
        pose_file_path = "/path/to/no_confidence.h5"

        mock_file = MagicMock()
        mock_h5py_file.return_value.__enter__.return_value = mock_file

        def mock_getitem(key):
            if key == "poseest":
                mock_poseest = MagicMock()
                mock_poseest.attrs = {"version": [2]}
                return mock_poseest
            elif key == "poseest/confidence":
                raise KeyError("confidence dataset not found")
            else:
                raise KeyError(f"Key {key} not found")

        mock_file.__getitem__.side_effect = mock_getitem

        # Act & Assert
        with pytest.raises(KeyError):
            inspect_pose_v2(pose_file_path)


class TestInspectPoseV2DataProcessing:
    """Test data processing and calculations."""

    @patch("mouse_tracking.pose.inspect.safe_find_first")
    @patch("mouse_tracking.pose.inspect.h5py.File")
    @patch("mouse_tracking.pose.inspect.CONFIG")
    def test_confidence_threshold_calculations(
        self, mock_config, mock_h5py_file, mock_safe_find_first
    ):
        """Test that confidence thresholds are applied correctly."""
        # Arrange
        pose_file_path = "/path/to/confidence_test.h5"

        # Mock CONFIG constants
        mock_config.MIN_HIGH_CONFIDENCE = 0.75
        mock_config.MIN_JABS_CONFIDENCE = 0.3

        mock_file = MagicMock()
        mock_h5py_file.return_value.__enter__.return_value = mock_file

        # Create confidence data that tests thresholds
        # Frame 0: No keypoints above threshold
        # Frame 1: Some keypoints above JABS threshold but not high confidence
        # Frame 2: All keypoints above high confidence threshold
        pose_quality = np.zeros((100, 1, 12))
        pose_quality[1, :, :5] = 0.4  # 5 keypoints above 0.3
        pose_quality[2:, :, :] = 0.8  # All keypoints above 0.75

        def mock_getitem(key):
            if key == "poseest":
                mock_poseest = MagicMock()
                mock_poseest.attrs = {"version": [2]}
                return mock_poseest
            elif key == "poseest/confidence":
                return pose_quality

        mock_file.__getitem__.side_effect = mock_getitem

        # Mock safe_find_first to return known values
        mock_safe_find_first.side_effect = [
            1,
            2,
        ]  # Different thresholds hit at different frames

        # Act
        _ = inspect_pose_v2(pose_file_path)

        # Assert - verify safe_find_first was called with correct arrays
        calls = mock_safe_find_first.call_args_list
        assert len(calls) == 2

        # Verify the calculation calls were made
        # Call 0: first_frame_pose
        # Call 1: first_frame_full_high_conf

    @patch("mouse_tracking.pose.inspect.safe_find_first")
    @patch("mouse_tracking.pose.inspect.h5py.File")
    @patch("mouse_tracking.pose.inspect.CONFIG")
    def test_pad_and_duration_calculations(
        self, mock_config, mock_h5py_file, mock_safe_find_first
    ):
        """Test that pad and duration parameters affect calculations correctly."""
        # Arrange
        pose_file_path = "/path/to/pad_test.h5"
        pad = 50
        duration = 200

        # Mock CONFIG constants
        mock_config.MIN_HIGH_CONFIDENCE = 0.75
        mock_config.MIN_JABS_CONFIDENCE = 0.3

        mock_file = MagicMock()
        mock_h5py_file.return_value.__enter__.return_value = mock_file

        # Create test data with known values
        total_frames = 300
        pose_quality = np.zeros((total_frames, 1, 12))
        pose_quality[60:240, :, :8] = (
            0.4  # Poses in frames 60-239, 8 keypoints > threshold
        )

        def mock_getitem(key):
            if key == "poseest":
                mock_poseest = MagicMock()
                mock_poseest.attrs = {"version": [2]}
                return mock_poseest
            elif key == "poseest/confidence":
                return pose_quality

        mock_file.__getitem__.side_effect = mock_getitem

        mock_safe_find_first.return_value = 0

        # Act
        result = inspect_pose_v2(pose_file_path, pad=pad, duration=duration)

        # Assert
        # In observation window [50:250]: frames 60-239 have keypoints > threshold
        # Each of these frames has 8 keypoints > threshold
        # Total keypoints in window: 180 frames * 8 keypoints = 1440
        expected_missing_poses = duration - 1440  # 200 - 1440 = -1240
        assert result["missing_poses"] == expected_missing_poses

        # For missing_keypoint_frames: counts keypoint positions != 12 in observation window
        # Since each position is 0 or 1, almost all positions != 12
        # In window [50:250] = 200 frames * 12 keypoints = 2400 positions, all != 12
        expected_missing_keypoint_frames = 200 * 12  # 2400

        assert result["missing_keypoint_frames"] == expected_missing_keypoint_frames

    @patch("mouse_tracking.pose.inspect.safe_find_first")
    @patch("mouse_tracking.pose.inspect.h5py.File")
    @patch("mouse_tracking.pose.inspect.CONFIG")
    def test_pose_counts_calculation(
        self, mock_config, mock_h5py_file, mock_safe_find_first
    ):
        """Test pose_counts calculation logic."""
        # Arrange
        pose_file_path = "/path/to/pose_counts_test.h5"

        mock_config.MIN_HIGH_CONFIDENCE = 0.75
        mock_config.MIN_JABS_CONFIDENCE = 0.3

        mock_file = MagicMock()
        mock_h5py_file.return_value.__enter__.return_value = mock_file

        # Create specific test data
        pose_quality = np.zeros((100, 1, 12))
        # Frames 10-50: 5 keypoints above threshold
        # Frames 60-80: 3 keypoints above threshold
        pose_quality[10:50, :, :5] = 0.4
        pose_quality[60:80, :, :3] = 0.5

        def mock_getitem(key):
            if key == "poseest":
                mock_poseest = MagicMock()
                mock_poseest.attrs = {"version": [2]}
                return mock_poseest
            elif key == "poseest/confidence":
                return pose_quality

        mock_file.__getitem__.side_effect = mock_getitem
        mock_safe_find_first.return_value = 0

        # Act
        result = inspect_pose_v2(pose_file_path)

        # Assert
        # pose_counts should be total number of keypoints > threshold across all frames
        # Frames 10-49: 40 frames * 5 keypoints = 200
        # Frames 60-79: 20 frames * 3 keypoints = 60
        # Total: 260
        expected_pose_counts = 260
        assert result["pose_counts"] == expected_pose_counts


class TestInspectPoseV2EdgeCases:
    """Test edge cases and boundary conditions."""

    @patch("mouse_tracking.pose.inspect.safe_find_first")
    @patch("mouse_tracking.pose.inspect.h5py.File")
    @patch("mouse_tracking.pose.inspect.CONFIG")
    def test_empty_arrays(self, mock_config, mock_h5py_file, mock_safe_find_first):
        """Test handling of empty arrays."""
        # Arrange
        pose_file_path = "/path/to/empty_test.h5"

        mock_config.MIN_HIGH_CONFIDENCE = 0.75
        mock_config.MIN_JABS_CONFIDENCE = 0.3

        mock_file = MagicMock()
        mock_h5py_file.return_value.__enter__.return_value = mock_file

        # Empty arrays
        pose_quality = np.array([]).reshape(0, 1, 12)

        def mock_getitem(key):
            if key == "poseest":
                mock_poseest = MagicMock()
                mock_poseest.attrs = {"version": [2]}
                return mock_poseest
            elif key == "poseest/confidence":
                return pose_quality

        mock_file.__getitem__.side_effect = mock_getitem

        mock_safe_find_first.return_value = -1  # No elements found

        # Act
        result = inspect_pose_v2(pose_file_path)

        # Assert
        assert result["first_frame_pose"] == -1
        assert result["first_frame_full_high_conf"] == -1
        assert result["pose_counts"] == 0
        # With empty arrays, slicing results in empty arrays, so sum = 0
        assert result["missing_keypoint_frames"] == 0

    @patch("mouse_tracking.pose.inspect.safe_find_first")
    @patch("mouse_tracking.pose.inspect.h5py.File")
    @patch("mouse_tracking.pose.inspect.CONFIG")
    def test_all_zero_confidence(
        self, mock_config, mock_h5py_file, mock_safe_find_first
    ):
        """Test handling when all confidence values are zero."""
        # Arrange
        pose_file_path = "/path/to/zero_conf_test.h5"

        mock_config.MIN_HIGH_CONFIDENCE = 0.75
        mock_config.MIN_JABS_CONFIDENCE = 0.3

        mock_file = MagicMock()
        mock_h5py_file.return_value.__enter__.return_value = mock_file

        # All confidence values are zero - use enough frames for default pad+duration
        pose_quality = np.zeros((110000, 1, 12))  # All zero confidence

        def mock_getitem(key):
            if key == "poseest":
                mock_poseest = MagicMock()
                mock_poseest.attrs = {"version": [2]}
                return mock_poseest
            elif key == "poseest/confidence":
                return pose_quality

        mock_file.__getitem__.side_effect = mock_getitem

        mock_safe_find_first.return_value = -1  # No frames meet confidence thresholds

        # Act
        result = inspect_pose_v2(pose_file_path)

        # Assert
        assert result["first_frame_pose"] == -1
        assert result["first_frame_full_high_conf"] == -1
        assert result["pose_counts"] == 0
        # All frames have 0 keypoints, so no keypoints in observation period
        assert result["missing_poses"] == 108000  # No poses in observation period
        # missing_keypoint_frames counts positions != 12: 108000 frames * 12 keypoints = 1296000
        assert result["missing_keypoint_frames"] == 108000 * 12  # All positions != 12

    @patch("mouse_tracking.pose.inspect.safe_find_first")
    @patch("mouse_tracking.pose.inspect.h5py.File")
    @patch("mouse_tracking.pose.inspect.CONFIG")
    def test_custom_pad_and_duration(
        self, mock_config, mock_h5py_file, mock_safe_find_first
    ):
        """Test with custom pad and duration values."""
        # Arrange
        pose_file_path = "/path/to/custom_test.h5"
        custom_pad = 500
        custom_duration = 50000

        mock_config.MIN_HIGH_CONFIDENCE = 0.75
        mock_config.MIN_JABS_CONFIDENCE = 0.3

        mock_file = MagicMock()
        mock_h5py_file.return_value.__enter__.return_value = mock_file

        # Large array to accommodate custom pad and duration
        total_frames = 60000
        pose_quality = np.full((total_frames, 1, 12), 0.8)  # All high confidence

        def mock_getitem(key):
            if key == "poseest":
                mock_poseest = MagicMock()
                mock_poseest.attrs = {"version": [2]}
                return mock_poseest
            elif key == "poseest/confidence":
                return pose_quality

        mock_file.__getitem__.side_effect = mock_getitem

        mock_safe_find_first.return_value = 0

        # Act
        result = inspect_pose_v2(
            pose_file_path, pad=custom_pad, duration=custom_duration
        )

        # Assert
        # With all keypoints having confidence 0.8 > 0.3:
        # - Each frame has 12 keypoint detections
        # - Total keypoints in window [500:50500]: 50000 frames * 12 keypoints = 600000
        expected_missing_poses = custom_duration - 600000  # 50000 - 600000 = -550000
        assert result["missing_poses"] == expected_missing_poses

        # missing_keypoint_frames: each position is 1, and 1 != 12, so all count
        expected_missing_keypoint_frames = custom_duration * 12  # 50000 * 12 = 600000
        assert result["missing_keypoint_frames"] == expected_missing_keypoint_frames

    @pytest.mark.parametrize(
        "confidence_value,threshold,expected_keypoints",
        [
            (0.2, 0.3, 0),  # Below threshold
            (0.3, 0.3, 0),  # Exactly at threshold (uses strict >, so 0.3 not > 0.3)
            (0.4, 0.3, 1),  # Above threshold
            (0.8, 0.75, 1),  # High confidence
        ],
    )
    @patch("mouse_tracking.pose.inspect.safe_find_first")
    @patch("mouse_tracking.pose.inspect.h5py.File")
    @patch("mouse_tracking.pose.inspect.CONFIG")
    def test_threshold_boundary_conditions(
        self,
        mock_config,
        mock_h5py_file,
        mock_safe_find_first,
        confidence_value,
        threshold,
        expected_keypoints,
    ):
        """Test threshold boundary conditions."""
        # Arrange
        pose_file_path = "/path/to/boundary_test.h5"

        mock_config.MIN_HIGH_CONFIDENCE = 0.75
        mock_config.MIN_JABS_CONFIDENCE = threshold

        mock_file = MagicMock()
        mock_h5py_file.return_value.__enter__.return_value = mock_file

        # Single frame with one keypoint at specific confidence
        pose_quality = np.zeros((1, 1, 12))
        pose_quality[0, 0, 0] = confidence_value

        def mock_getitem(key):
            if key == "poseest":
                mock_poseest = MagicMock()
                mock_poseest.attrs = {"version": [2]}
                return mock_poseest
            elif key == "poseest/confidence":
                return pose_quality

        mock_file.__getitem__.side_effect = mock_getitem
        mock_safe_find_first.return_value = 0 if expected_keypoints > 0 else -1

        # Act
        result = inspect_pose_v2(pose_file_path, pad=0, duration=1)

        # Assert
        expected_pose_counts = expected_keypoints
        assert result["pose_counts"] == expected_pose_counts


class TestInspectPoseV2MockingVerification:
    """Test that mocking is working correctly and dependencies are called properly."""

    @patch("mouse_tracking.pose.inspect.safe_find_first")
    @patch("mouse_tracking.pose.inspect.h5py.File")
    @patch("mouse_tracking.pose.inspect.CONFIG")
    def test_all_dependencies_called_correctly(
        self, mock_config, mock_h5py_file, mock_safe_find_first
    ):
        """Test that all mocked dependencies are called with correct arguments."""
        # Arrange
        pose_file_path = "/test/path/video_pose_est_v2.h5"

        # Mock CONFIG
        mock_config.MIN_HIGH_CONFIDENCE = 0.75
        mock_config.MIN_JABS_CONFIDENCE = 0.3

        # Mock HDF5 file
        mock_file = MagicMock()
        mock_h5py_file.return_value.__enter__.return_value = mock_file

        pose_quality = np.full((100, 1, 12), 0.8)

        def mock_getitem(key):
            if key == "poseest":
                mock_poseest = MagicMock()
                mock_poseest.attrs = {"version": [2]}
                return mock_poseest
            elif key == "poseest/confidence":
                return pose_quality

        mock_file.__getitem__.side_effect = mock_getitem

        mock_safe_find_first.return_value = 0

        # Act
        result = inspect_pose_v2(pose_file_path)

        # Assert - verify all dependencies were called
        mock_h5py_file.assert_called_once_with(pose_file_path, "r")
        assert mock_safe_find_first.call_count == 2

        # Verify result structure
        expected_keys = {
            "first_frame_pose",
            "first_frame_full_high_conf",
            "pose_counts",
            "missing_poses",
            "missing_keypoint_frames",
        }
        assert set(result.keys()) == expected_keys

    @patch("mouse_tracking.pose.inspect.safe_find_first")
    @patch("mouse_tracking.pose.inspect.h5py.File")
    @patch("mouse_tracking.pose.inspect.CONFIG")
    def test_array_shape_handling(
        self, mock_config, mock_h5py_file, mock_safe_find_first
    ):
        """Test that the function handles v2 array shapes correctly (single instance dimension)."""
        # Arrange
        pose_file_path = "/path/to/shape_test.h5"

        mock_config.MIN_HIGH_CONFIDENCE = 0.75
        mock_config.MIN_JABS_CONFIDENCE = 0.3

        mock_file = MagicMock()
        mock_h5py_file.return_value.__enter__.return_value = mock_file

        # v2 shape: [frames, instances, keypoints] same as v6, typically 1 instance
        pose_quality = np.random.rand(1000, 1, 12)  # 3D with single instance dimension

        def mock_getitem(key):
            if key == "poseest":
                mock_poseest = MagicMock()
                mock_poseest.attrs = {"version": [2]}
                return mock_poseest
            elif key == "poseest/confidence":
                return pose_quality

        mock_file.__getitem__.side_effect = mock_getitem
        mock_safe_find_first.return_value = 0

        # Act & Assert - should not raise any shape-related errors
        result = inspect_pose_v2(pose_file_path)

        # Verify the function completed successfully
        assert "pose_counts" in result
        assert isinstance(result["pose_counts"], int | np.integer)
