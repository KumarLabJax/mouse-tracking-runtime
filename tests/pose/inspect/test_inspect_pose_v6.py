"""
Unit tests for the inspect_pose_v6 function.

This module provides comprehensive test coverage for the inspect_pose_v6 function,
including success paths, error conditions, and edge cases with properly mocked
dependencies to ensure backwards compatibility testing.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mouse_tracking.pose.inspect import inspect_pose_v6


class TestInspectPoseV6BasicFunctionality:
    """Test basic functionality of inspect_pose_v6."""

    @patch("mouse_tracking.pose.inspect.hash_file")
    @patch("mouse_tracking.pose.inspect.safe_find_first")
    @patch("mouse_tracking.pose.inspect.h5py.File")
    @patch("mouse_tracking.pose.inspect.CONFIG")
    def test_successful_inspection_with_corners(
        self, mock_config, mock_h5py_file, mock_safe_find_first, mock_hash_file
    ):
        """Test successful inspection of a valid v6 pose file with corners present."""
        # Arrange
        pose_file_path = "/path/to/test/folder1/folder2/video_pose_est_v6.h5"
        pad = 150
        duration = 108000

        # Mock CONFIG constants
        mock_config.MIN_HIGH_CONFIDENCE = 0.75
        mock_config.MIN_JABS_CONFIDENCE = 0.3
        mock_config.MIN_JABS_KEYPOINTS = 3
        mock_config.MIN_GAIT_CONFIDENCE = 0.3
        mock_config.BASE_TAIL_INDEX = 9
        mock_config.LEFT_REAR_PAW_INDEX = 7
        mock_config.RIGHT_REAR_PAW_INDEX = 8

        # Mock HDF5 file structure
        mock_file = MagicMock()
        mock_h5py_file.return_value.__enter__.return_value = mock_file

        # Mock version check
        mock_file.__getitem__.return_value.attrs.__getitem__.return_value = [6]

        # Create test data arrays
        num_frames = 110000
        pose_counts = np.zeros(num_frames, dtype=np.uint8)
        pose_counts[100:105000] = 1  # Poses present from frame 100

        pose_quality = np.random.rand(num_frames, 1, 12)
        pose_quality[:100] = 0  # No confidence before frame 100
        pose_quality[100:110000] = 0.8  # High confidence after frame 100

        pose_tracks = np.zeros((num_frames, 1), dtype=np.uint32)
        pose_tracks[100:50000, 0] = 1  # First tracklet
        pose_tracks[50000:105000, 0] = 2  # Second tracklet

        seg_ids = np.zeros(num_frames, dtype=np.uint32)
        seg_ids[150:105000] = 1  # Segmentation starts at frame 150

        # Mock dataset access
        def mock_getitem(key):
            if key == "poseest":
                mock_poseest = MagicMock()
                mock_poseest.attrs = {"version": [6]}
                return mock_poseest
            elif key == "poseest/instance_count":
                return pose_counts
            elif key == "poseest/confidence":
                return pose_quality
            elif key == "poseest/instance_track_id":
                return pose_tracks
            elif key == "poseest/longterm_seg_id":
                return seg_ids
            else:
                raise KeyError(f"Key {key} not found")

        mock_file.__getitem__.side_effect = mock_getitem
        mock_file.__contains__.side_effect = lambda key: key == "static_objects/corners"

        # Mock safe_find_first to return sequential values for testing
        mock_safe_find_first.side_effect = [
            100,
            100,
            100,
            100,
            150,
        ]  # Different first frames

        # Mock hash_file
        mock_hash_file.return_value = "abcdef123456"

        # Act
        result = inspect_pose_v6(pose_file_path, pad=pad, duration=duration)

        # Assert
        assert result["pose_file"] == "video_pose_est_v6.h5"
        assert result["pose_hash"] == "abcdef123456"
        assert result["video_name"] == "folder1/folder2/video"
        assert result["video_duration"] == num_frames
        assert result["corners_present"] is True
        assert result["first_frame_pose"] == 100
        assert result["first_frame_full_high_conf"] == 100
        assert result["first_frame_jabs"] == 100
        assert result["first_frame_gait"] == 100
        assert result["first_frame_seg"] == 150
        assert result["pose_counts"] == np.sum(pose_counts)
        assert result["seg_counts"] == np.sum(seg_ids > 0)
        assert result["missing_poses"] == duration - np.sum(
            pose_counts[pad : pad + duration]
        )
        assert result["missing_segs"] == duration - np.sum(
            seg_ids[pad : pad + duration] > 0
        )

        # Verify mocked functions were called correctly
        mock_hash_file.assert_called_once()
        assert mock_safe_find_first.call_count == 5
        mock_h5py_file.assert_called_once_with(pose_file_path, "r")

    @patch("mouse_tracking.pose.inspect.hash_file")
    @patch("mouse_tracking.pose.inspect.safe_find_first")
    @patch("mouse_tracking.pose.inspect.h5py.File")
    @patch("mouse_tracking.pose.inspect.CONFIG")
    def test_successful_inspection_without_corners(
        self, mock_config, mock_h5py_file, mock_safe_find_first, mock_hash_file
    ):
        """Test successful inspection of a valid v6 pose file without corners."""
        # Arrange
        pose_file_path = "/path/to/test_video_pose_est_v6.h5"

        # Mock CONFIG constants
        mock_config.MIN_HIGH_CONFIDENCE = 0.75
        mock_config.MIN_JABS_CONFIDENCE = 0.3
        mock_config.MIN_JABS_KEYPOINTS = 3
        mock_config.MIN_GAIT_CONFIDENCE = 0.3
        mock_config.BASE_TAIL_INDEX = 9
        mock_config.LEFT_REAR_PAW_INDEX = 7
        mock_config.RIGHT_REAR_PAW_INDEX = 8

        # Mock HDF5 file structure
        mock_file = MagicMock()
        mock_h5py_file.return_value.__enter__.return_value = mock_file

        # Create minimal test data
        pose_counts = np.ones(1000, dtype=np.uint8)
        pose_quality = np.full((1000, 1, 12), 0.8)
        pose_tracks = np.ones((1000, 1), dtype=np.uint32)
        seg_ids = np.ones(1000, dtype=np.uint32)

        def mock_getitem(key):
            if key == "poseest":
                mock_poseest = MagicMock()
                mock_poseest.attrs = {"version": [6]}
                return mock_poseest
            elif key == "poseest/instance_count":
                return pose_counts
            elif key == "poseest/confidence":
                return pose_quality
            elif key == "poseest/instance_track_id":
                return pose_tracks
            elif key == "poseest/longterm_seg_id":
                return seg_ids

        mock_file.__getitem__.side_effect = mock_getitem
        mock_file.__contains__.return_value = False  # No corners

        mock_safe_find_first.return_value = 0
        mock_hash_file.return_value = "xyz789"

        # Act
        result = inspect_pose_v6(pose_file_path)

        # Assert
        assert result["corners_present"] is False
        assert result["video_name"] == "path/to/test_video"


class TestInspectPoseV6ErrorHandling:
    """Test error handling scenarios."""

    @patch("mouse_tracking.pose.inspect.h5py.File")
    def test_version_less_than_6_raises_error(self, mock_h5py_file):
        """Test that version < 6 raises ValueError."""
        # Arrange
        pose_file_path = "/path/to/test_v5.h5"

        mock_file = MagicMock()
        mock_h5py_file.return_value.__enter__.return_value = mock_file

        # Mock version 5
        mock_poseest = MagicMock()
        mock_poseest.attrs = {"version": [5]}
        mock_file.__getitem__.return_value = mock_poseest

        # Act & Assert
        with pytest.raises(
            ValueError, match=r"Only v6\+ pose files are supported.*version 5"
        ):
            inspect_pose_v6(pose_file_path)

    @patch("mouse_tracking.pose.inspect.h5py.File")
    def test_multiple_instances_raises_error(self, mock_h5py_file):
        """Test that multiple instances raises ValueError."""
        # Arrange
        pose_file_path = "/path/to/multi_mouse.h5"

        mock_file = MagicMock()
        mock_h5py_file.return_value.__enter__.return_value = mock_file

        # Mock multi-mouse data with non-empty array to avoid max() error
        pose_counts = np.array([2, 1, 3, 1])  # Max is 3 > 1

        def mock_getitem(key):
            if key == "poseest":
                mock_poseest = MagicMock()
                mock_poseest.attrs = {"version": [6]}
                return mock_poseest
            elif key == "poseest/instance_count":
                return pose_counts

        mock_file.__getitem__.side_effect = mock_getitem

        # Act & Assert
        with pytest.raises(
            ValueError,
            match="Only single mouse pose files are supported.*contains multiple instances",
        ):
            inspect_pose_v6(pose_file_path)

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
            inspect_pose_v6(pose_file_path)


class TestInspectPoseV6DataProcessing:
    """Test data processing and calculations."""

    @patch("mouse_tracking.pose.inspect.hash_file")
    @patch("mouse_tracking.pose.inspect.safe_find_first")
    @patch("mouse_tracking.pose.inspect.h5py.File")
    @patch("mouse_tracking.pose.inspect.CONFIG")
    def test_confidence_threshold_calculations(
        self, mock_config, mock_h5py_file, mock_safe_find_first, mock_hash_file
    ):
        """Test that confidence thresholds are applied correctly."""
        # Arrange
        pose_file_path = "/path/to/confidence_test.h5"

        # Mock CONFIG constants
        mock_config.MIN_HIGH_CONFIDENCE = 0.75
        mock_config.MIN_JABS_CONFIDENCE = 0.3
        mock_config.MIN_JABS_KEYPOINTS = 3
        mock_config.MIN_GAIT_CONFIDENCE = 0.3
        mock_config.BASE_TAIL_INDEX = 9
        mock_config.LEFT_REAR_PAW_INDEX = 7
        mock_config.RIGHT_REAR_PAW_INDEX = 8

        mock_file = MagicMock()
        mock_h5py_file.return_value.__enter__.return_value = mock_file

        # Create confidence data that tests thresholds
        pose_counts = np.ones(100, dtype=np.uint8)

        # Frame 0: No keypoints above threshold
        # Frame 1: Some keypoints above JABS threshold but not high confidence
        # Frame 2: All keypoints above high confidence threshold
        pose_quality = np.zeros((100, 1, 12))
        pose_quality[1, 0, :5] = 0.4  # 5 keypoints above 0.3
        pose_quality[2:, 0, :] = 0.8  # All keypoints above 0.75

        pose_tracks = np.ones((100, 1), dtype=np.uint32)
        seg_ids = np.ones(100, dtype=np.uint32)

        def mock_getitem(key):
            if key == "poseest":
                mock_poseest = MagicMock()
                mock_poseest.attrs = {"version": [6]}
                return mock_poseest
            elif key == "poseest/instance_count":
                return pose_counts
            elif key == "poseest/confidence":
                return pose_quality
            elif key == "poseest/instance_track_id":
                return pose_tracks
            elif key == "poseest/longterm_seg_id":
                return seg_ids

        mock_file.__getitem__.side_effect = mock_getitem
        mock_file.__contains__.return_value = True

        # Mock safe_find_first to return known values
        mock_safe_find_first.side_effect = [
            0,
            2,
            1,
            2,
            0,
        ]  # Different thresholds hit at different frames
        mock_hash_file.return_value = "test_hash"

        # Act
        _ = inspect_pose_v6(pose_file_path)

        # Assert - verify safe_find_first was called with correct arrays
        calls = mock_safe_find_first.call_args_list
        assert len(calls) == 5

        # Verify the calculation calls were made with proper arrays
        # Call 0: pose_counts > 0
        # Call 1: high_conf_keypoints (all confidence > 0.75)
        # Call 2: jabs_keypoints >= MIN_JABS_KEYPOINTS
        # Call 3: gait_keypoints (specific keypoints > 0.3)
        # Call 4: seg_ids > 0

    @patch("mouse_tracking.pose.inspect.hash_file")
    @patch("mouse_tracking.pose.inspect.safe_find_first")
    @patch("mouse_tracking.pose.inspect.h5py.File")
    @patch("mouse_tracking.pose.inspect.CONFIG")
    def test_pad_and_duration_calculations(
        self, mock_config, mock_h5py_file, mock_safe_find_first, mock_hash_file
    ):
        """Test that pad and duration parameters affect calculations correctly."""
        # Arrange
        pose_file_path = "/path/to/pad_test.h5"
        pad = 50
        duration = 200

        # Mock CONFIG constants
        mock_config.MIN_HIGH_CONFIDENCE = 0.75
        mock_config.MIN_JABS_CONFIDENCE = 0.3
        mock_config.MIN_JABS_KEYPOINTS = 3
        mock_config.MIN_GAIT_CONFIDENCE = 0.3
        mock_config.BASE_TAIL_INDEX = 9
        mock_config.LEFT_REAR_PAW_INDEX = 7
        mock_config.RIGHT_REAR_PAW_INDEX = 8

        mock_file = MagicMock()
        mock_h5py_file.return_value.__enter__.return_value = mock_file

        # Create test data with known values
        total_frames = 300
        pose_counts = np.zeros(total_frames, dtype=np.uint8)
        pose_counts[60:240] = 1  # Poses in frames 60-239 (180 frames)

        pose_quality = np.full((total_frames, 1, 12), 0.8)
        pose_tracks = np.ones((total_frames, 1), dtype=np.uint32)

        seg_ids = np.zeros(total_frames, dtype=np.uint32)
        seg_ids[70:230] = 1  # Segmentation in frames 70-229 (160 frames)

        def mock_getitem(key):
            if key == "poseest":
                mock_poseest = MagicMock()
                mock_poseest.attrs = {"version": [6]}
                return mock_poseest
            elif key == "poseest/instance_count":
                return pose_counts
            elif key == "poseest/confidence":
                return pose_quality
            elif key == "poseest/instance_track_id":
                return pose_tracks
            elif key == "poseest/longterm_seg_id":
                return seg_ids

        mock_file.__getitem__.side_effect = mock_getitem
        mock_file.__contains__.return_value = False

        mock_safe_find_first.return_value = 0
        mock_hash_file.return_value = "pad_test_hash"

        # Act
        result = inspect_pose_v6(pose_file_path, pad=pad, duration=duration)

        # Assert
        # Total poses in observation window (frames 50-249, but poses only in 60-239)
        poses_in_window = np.sum(pose_counts[50:250])  # Should be 180
        missing_poses = duration - poses_in_window  # 200 - 180 = 20

        # Total segmentations in observation window (frames 50-249, but seg only in 70-229)
        segs_in_window = np.sum(seg_ids[50:250] > 0)  # Should be 160
        missing_segs = duration - segs_in_window  # 200 - 160 = 40

        assert result["missing_poses"] == missing_poses
        assert result["missing_segs"] == missing_segs

    @patch("mouse_tracking.pose.inspect.hash_file")
    @patch("mouse_tracking.pose.inspect.safe_find_first")
    @patch("mouse_tracking.pose.inspect.h5py.File")
    @patch("mouse_tracking.pose.inspect.CONFIG")
    def test_tracklet_calculation(
        self, mock_config, mock_h5py_file, mock_safe_find_first, mock_hash_file
    ):
        """Test tracklet counting in observation duration."""
        # Arrange
        pose_file_path = "/path/to/tracklet_test.h5"
        pad = 10
        duration = 100

        # Mock CONFIG constants
        mock_config.MIN_HIGH_CONFIDENCE = 0.75
        mock_config.MIN_JABS_CONFIDENCE = 0.3
        mock_config.MIN_JABS_KEYPOINTS = 3
        mock_config.MIN_GAIT_CONFIDENCE = 0.3
        mock_config.BASE_TAIL_INDEX = 9
        mock_config.LEFT_REAR_PAW_INDEX = 7
        mock_config.RIGHT_REAR_PAW_INDEX = 8

        mock_file = MagicMock()
        mock_h5py_file.return_value.__enter__.return_value = mock_file

        # Create tracklet data
        total_frames = 150
        pose_counts = np.ones(total_frames, dtype=np.uint8)

        pose_tracks = np.zeros((total_frames, 1), dtype=np.uint32)
        # Tracklet 1: frames 15-50
        pose_tracks[15:51, 0] = 1
        # Tracklet 2: frames 60-90
        pose_tracks[60:91, 0] = 2
        # Tracklet 3: frames 100-120
        pose_tracks[100:121, 0] = 3

        pose_quality = np.full((total_frames, 1, 12), 0.8)
        seg_ids = np.ones(total_frames, dtype=np.uint32)

        def mock_getitem(key):
            if key == "poseest":
                mock_poseest = MagicMock()
                mock_poseest.attrs = {"version": [6]}
                return mock_poseest
            elif key == "poseest/instance_count":
                return pose_counts
            elif key == "poseest/confidence":
                return pose_quality
            elif key == "poseest/instance_track_id":
                return pose_tracks
            elif key == "poseest/longterm_seg_id":
                return seg_ids

        mock_file.__getitem__.side_effect = mock_getitem
        mock_file.__contains__.return_value = True

        mock_safe_find_first.return_value = 0
        mock_hash_file.return_value = "tracklet_hash"

        # Act
        result = inspect_pose_v6(pose_file_path, pad=pad, duration=duration)

        # Assert
        # In observation window (frames 10-109):
        # Tracklet 0: frames 10-14, 51-59, 91-99 (gaps between other tracklets)
        # Tracklet 1: frames 15-50 (partially in window)
        # Tracklet 2: frames 60-90 (fully in window)
        # Tracklet 3: frames 100-109 (partially in window)
        # Should count 4 unique tracklets (including tracklet 0 for gaps)
        assert result["pose_tracklets"] == 4


class TestInspectPoseV6VideoNameParsing:
    """Test video name parsing logic."""

    @pytest.mark.parametrize(
        "pose_file_path,expected_video_name",
        [
            # Standard cases
            ("/folder1/folder2/video_pose_est_v6.h5", "folder1/folder2/video"),
            ("/a/b/test_video_pose_est_v6.h5", "a/b/test_video"),
            ("/x/y/z/sample_pose_est_v10.h5", "y/z/sample"),
            # Edge cases
            ("/single_folder/file_pose_est_v6.h5", "//single_folder/file"),
            ("/file_pose_est_v6.h5", "//file"),
            ("/a/b/c/d/e/long_path_pose_est_v6.h5", "d/e/long_path"),
            # Different version numbers
            ("/folder1/folder2/video_pose_est_v2.h5", "folder1/folder2/video"),
            ("/folder1/folder2/video_pose_est_v15.h5", "folder1/folder2/video"),
        ],
    )
    @patch("mouse_tracking.pose.inspect.hash_file")
    @patch("mouse_tracking.pose.inspect.safe_find_first")
    @patch("mouse_tracking.pose.inspect.h5py.File")
    @patch("mouse_tracking.pose.inspect.CONFIG")
    def test_video_name_parsing(
        self,
        mock_config,
        mock_h5py_file,
        mock_safe_find_first,
        mock_hash_file,
        pose_file_path,
        expected_video_name,
    ):
        """Test video name parsing from file path."""
        # Arrange
        mock_config.MIN_HIGH_CONFIDENCE = 0.75
        mock_config.MIN_JABS_CONFIDENCE = 0.3
        mock_config.MIN_JABS_KEYPOINTS = 3
        mock_config.MIN_GAIT_CONFIDENCE = 0.3
        mock_config.BASE_TAIL_INDEX = 9
        mock_config.LEFT_REAR_PAW_INDEX = 7
        mock_config.RIGHT_REAR_PAW_INDEX = 8

        mock_file = MagicMock()
        mock_h5py_file.return_value.__enter__.return_value = mock_file

        # Mock minimal valid data
        pose_counts = np.ones(100, dtype=np.uint8)
        pose_quality = np.full((100, 1, 12), 0.8)
        pose_tracks = np.ones((100, 1), dtype=np.uint32)
        seg_ids = np.ones(100, dtype=np.uint32)

        def mock_getitem(key):
            if key == "poseest":
                mock_poseest = MagicMock()
                mock_poseest.attrs = {"version": [6]}
                return mock_poseest
            elif key == "poseest/instance_count":
                return pose_counts
            elif key == "poseest/confidence":
                return pose_quality
            elif key == "poseest/instance_track_id":
                return pose_tracks
            elif key == "poseest/longterm_seg_id":
                return seg_ids

        mock_file.__getitem__.side_effect = mock_getitem
        mock_file.__contains__.return_value = True

        mock_safe_find_first.return_value = 0
        mock_hash_file.return_value = "test_hash"

        # Act
        result = inspect_pose_v6(pose_file_path)

        # Assert
        assert result["video_name"] == expected_video_name


class TestInspectPoseV6EdgeCases:
    """Test edge cases and boundary conditions."""

    @patch("mouse_tracking.pose.inspect.hash_file")
    @patch("mouse_tracking.pose.inspect.safe_find_first")
    @patch("mouse_tracking.pose.inspect.h5py.File")
    @patch("mouse_tracking.pose.inspect.CONFIG")
    def test_empty_arrays(
        self, mock_config, mock_h5py_file, mock_safe_find_first, mock_hash_file
    ):
        """Test handling of empty arrays - this should raise ValueError due to np.max on empty array."""
        # Arrange
        pose_file_path = "/path/to/empty_test.h5"

        mock_config.MIN_HIGH_CONFIDENCE = 0.75
        mock_config.MIN_JABS_CONFIDENCE = 0.3
        mock_config.MIN_JABS_KEYPOINTS = 3
        mock_config.MIN_GAIT_CONFIDENCE = 0.3
        mock_config.BASE_TAIL_INDEX = 9
        mock_config.LEFT_REAR_PAW_INDEX = 7
        mock_config.RIGHT_REAR_PAW_INDEX = 8

        mock_file = MagicMock()
        mock_h5py_file.return_value.__enter__.return_value = mock_file

        # Empty arrays
        pose_counts = np.array([], dtype=np.uint8)
        pose_quality = np.array([]).reshape(0, 1, 12)
        pose_tracks = np.array([]).reshape(0, 1)
        seg_ids = np.array([], dtype=np.uint32)

        def mock_getitem(key):
            if key == "poseest":
                mock_poseest = MagicMock()
                mock_poseest.attrs = {"version": [6]}
                return mock_poseest
            elif key == "poseest/instance_count":
                return pose_counts
            elif key == "poseest/confidence":
                return pose_quality
            elif key == "poseest/instance_track_id":
                return pose_tracks
            elif key == "poseest/longterm_seg_id":
                return seg_ids

        mock_file.__getitem__.side_effect = mock_getitem
        mock_file.__contains__.return_value = False

        mock_safe_find_first.return_value = -1  # No elements found
        mock_hash_file.return_value = "empty_hash"

        # Act & Assert
        # The function should raise ValueError when calling np.max on empty pose_counts array
        with pytest.raises(
            ValueError, match="zero-size array to reduction operation maximum"
        ):
            inspect_pose_v6(pose_file_path)

    @patch("mouse_tracking.pose.inspect.hash_file")
    @patch("mouse_tracking.pose.inspect.safe_find_first")
    @patch("mouse_tracking.pose.inspect.h5py.File")
    @patch("mouse_tracking.pose.inspect.CONFIG")
    def test_all_zero_confidence(
        self, mock_config, mock_h5py_file, mock_safe_find_first, mock_hash_file
    ):
        """Test handling when all confidence values are zero."""
        # Arrange
        pose_file_path = "/path/to/zero_conf_test.h5"

        mock_config.MIN_HIGH_CONFIDENCE = 0.75
        mock_config.MIN_JABS_CONFIDENCE = 0.3
        mock_config.MIN_JABS_KEYPOINTS = 3
        mock_config.MIN_GAIT_CONFIDENCE = 0.3
        mock_config.BASE_TAIL_INDEX = 9
        mock_config.LEFT_REAR_PAW_INDEX = 7
        mock_config.RIGHT_REAR_PAW_INDEX = 8

        mock_file = MagicMock()
        mock_h5py_file.return_value.__enter__.return_value = mock_file

        # All confidence values are zero - use enough frames for default pad+duration
        pose_counts = np.ones(110000, dtype=np.uint8)
        pose_quality = np.zeros((110000, 1, 12))  # All zero confidence
        pose_tracks = np.ones((110000, 1), dtype=np.uint32)
        seg_ids = np.ones(110000, dtype=np.uint32)

        def mock_getitem(key):
            if key == "poseest":
                mock_poseest = MagicMock()
                mock_poseest.attrs = {"version": [6]}
                return mock_poseest
            elif key == "poseest/instance_count":
                return pose_counts
            elif key == "poseest/confidence":
                return pose_quality
            elif key == "poseest/instance_track_id":
                return pose_tracks
            elif key == "poseest/longterm_seg_id":
                return seg_ids

        mock_file.__getitem__.side_effect = mock_getitem
        mock_file.__contains__.return_value = True

        mock_safe_find_first.return_value = -1  # No frames meet confidence thresholds
        mock_hash_file.return_value = "zero_conf_hash"

        # Act
        result = inspect_pose_v6(pose_file_path)

        # Assert
        assert result["first_frame_full_high_conf"] == -1
        assert result["first_frame_jabs"] == -1
        assert result["first_frame_gait"] == -1
        # With all zero confidence, num_keypoints = 12 - 12 = 0, so all frames != 12
        # Default duration is 108000, so all frames in observation period are missing keypoints
        assert (
            result["missing_keypoint_frames"] == 108000
        )  # All frames in observation period missing keypoints

    @patch("mouse_tracking.pose.inspect.hash_file")
    @patch("mouse_tracking.pose.inspect.safe_find_first")
    @patch("mouse_tracking.pose.inspect.h5py.File")
    @patch("mouse_tracking.pose.inspect.CONFIG")
    def test_custom_pad_and_duration(
        self, mock_config, mock_h5py_file, mock_safe_find_first, mock_hash_file
    ):
        """Test with custom pad and duration values."""
        # Arrange
        pose_file_path = "/path/to/custom_test.h5"
        custom_pad = 500
        custom_duration = 50000

        mock_config.MIN_HIGH_CONFIDENCE = 0.75
        mock_config.MIN_JABS_CONFIDENCE = 0.3
        mock_config.MIN_JABS_KEYPOINTS = 3
        mock_config.MIN_GAIT_CONFIDENCE = 0.3
        mock_config.BASE_TAIL_INDEX = 9
        mock_config.LEFT_REAR_PAW_INDEX = 7
        mock_config.RIGHT_REAR_PAW_INDEX = 8

        mock_file = MagicMock()
        mock_h5py_file.return_value.__enter__.return_value = mock_file

        # Large array to accommodate custom pad and duration
        total_frames = 60000
        pose_counts = np.ones(total_frames, dtype=np.uint8)
        pose_quality = np.full((total_frames, 1, 12), 0.8)
        pose_tracks = np.ones((total_frames, 1), dtype=np.uint32)
        seg_ids = np.ones(total_frames, dtype=np.uint32)

        def mock_getitem(key):
            if key == "poseest":
                mock_poseest = MagicMock()
                mock_poseest.attrs = {"version": [6]}
                return mock_poseest
            elif key == "poseest/instance_count":
                return pose_counts
            elif key == "poseest/confidence":
                return pose_quality
            elif key == "poseest/instance_track_id":
                return pose_tracks
            elif key == "poseest/longterm_seg_id":
                return seg_ids

        mock_file.__getitem__.side_effect = mock_getitem
        mock_file.__contains__.return_value = True

        mock_safe_find_first.return_value = 0
        mock_hash_file.return_value = "custom_hash"

        # Act
        result = inspect_pose_v6(
            pose_file_path, pad=custom_pad, duration=custom_duration
        )

        # Assert
        # With all frames having poses/segs, missing should be 0
        assert result["missing_poses"] == 0
        assert result["missing_segs"] == 0
        # Keypoints calculation: 12 - sum of zeros = 12 for all frames
        assert result["missing_keypoint_frames"] == 0


class TestInspectPoseV6MockingVerification:
    """Test that mocking is working correctly and dependencies are called properly."""

    @patch("mouse_tracking.pose.inspect.hash_file")
    @patch("mouse_tracking.pose.inspect.safe_find_first")
    @patch("mouse_tracking.pose.inspect.h5py.File")
    @patch("mouse_tracking.pose.inspect.CONFIG")
    @patch("mouse_tracking.pose.inspect.Path")
    @patch("mouse_tracking.pose.inspect.re.sub")
    def test_all_dependencies_called_correctly(
        self,
        mock_re_sub,
        mock_path,
        mock_config,
        mock_h5py_file,
        mock_safe_find_first,
        mock_hash_file,
    ):
        """Test that all mocked dependencies are called with correct arguments."""
        # Arrange
        pose_file_path = "/test/path/video_pose_est_v6.h5"

        # Mock CONFIG
        mock_config.MIN_HIGH_CONFIDENCE = 0.75
        mock_config.MIN_JABS_CONFIDENCE = 0.3
        mock_config.MIN_JABS_KEYPOINTS = 3
        mock_config.MIN_GAIT_CONFIDENCE = 0.3
        mock_config.BASE_TAIL_INDEX = 9
        mock_config.LEFT_REAR_PAW_INDEX = 7
        mock_config.RIGHT_REAR_PAW_INDEX = 8

        # Mock Path operations
        mock_path_instance = MagicMock()
        mock_path_instance.name = "video_pose_est_v6.h5"
        mock_path_instance.stem = "video_pose_est_v6"
        mock_path_instance.parts = ("/", "test", "path", "video_pose_est_v6.h5")
        mock_path.return_value = mock_path_instance

        # Mock regex substitution
        mock_re_sub.return_value = "video"

        # Mock HDF5 file
        mock_file = MagicMock()
        mock_h5py_file.return_value.__enter__.return_value = mock_file

        pose_counts = np.ones(100, dtype=np.uint8)
        pose_quality = np.full((100, 1, 12), 0.8)
        pose_tracks = np.ones((100, 1), dtype=np.uint32)
        seg_ids = np.ones(100, dtype=np.uint32)

        def mock_getitem(key):
            if key == "poseest":
                mock_poseest = MagicMock()
                mock_poseest.attrs = {"version": [6]}
                return mock_poseest
            elif key == "poseest/instance_count":
                return pose_counts
            elif key == "poseest/confidence":
                return pose_quality
            elif key == "poseest/instance_track_id":
                return pose_tracks
            elif key == "poseest/longterm_seg_id":
                return seg_ids

        mock_file.__getitem__.side_effect = mock_getitem
        mock_file.__contains__.return_value = True

        mock_safe_find_first.return_value = 0
        mock_hash_file.return_value = "dependency_test_hash"

        # Act
        result = inspect_pose_v6(pose_file_path)

        # Assert - verify all dependencies were called
        mock_h5py_file.assert_called_once_with(pose_file_path, "r")
        mock_hash_file.assert_called_once()
        assert mock_safe_find_first.call_count == 5
        mock_path.assert_called()
        mock_re_sub.assert_called_once_with(
            "_pose_est_v[0-9]+", "", "video_pose_est_v6"
        )

        # Verify result structure
        expected_keys = {
            "pose_file",
            "pose_hash",
            "video_name",
            "video_duration",
            "corners_present",
            "first_frame_pose",
            "first_frame_full_high_conf",
            "first_frame_jabs",
            "first_frame_gait",
            "first_frame_seg",
            "pose_counts",
            "seg_counts",
            "missing_poses",
            "missing_segs",
            "pose_tracklets",
            "missing_keypoint_frames",
        }
        assert set(result.keys()) == expected_keys
