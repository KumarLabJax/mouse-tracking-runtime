"""Comprehensive unit tests for the promote_pose_data function."""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from mouse_tracking.utils.writers import promote_pose_data


class TestPromotePoseDataV2ToV3:
    """Test v2 to v3 promotion functionality."""

    @patch("mouse_tracking.utils.writers.write_pose_v3_data")
    @patch("mouse_tracking.utils.writers.write_pose_v2_data")
    @patch("mouse_tracking.utils.writers.convert_v2_to_v3")
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_v2_to_v3_basic_promotion(
        self,
        mock_h5py_file,
        mock_convert_v2_to_v3,
        mock_write_pose_v2_data,
        mock_write_pose_v3_data,
    ):
        """Test basic v2 to v3 promotion with config and model attributes."""
        # Arrange
        pose_file = "test_pose.h5"
        current_version = 2
        new_version = 3

        # Mock HDF5 file data
        mock_file_context = MagicMock()
        mock_h5py_file.return_value.__enter__.return_value = mock_file_context

        # Mock pose and confidence data
        original_pose_data = np.random.rand(10, 12, 2).astype(np.float32)
        original_conf_data = np.random.rand(10, 12).astype(np.float32)
        mock_file_context.__getitem__.side_effect = lambda key: {
            "poseest/points": Mock(
                __getitem__=lambda self, slice_obj: original_pose_data,
                attrs={"config": "test_config", "model": "test_model"},
            ),
            "poseest/confidence": Mock(
                __getitem__=lambda self, slice_obj: original_conf_data
            ),
        }[key]

        # Mock convert_v2_to_v3 return values
        converted_pose_data = np.random.rand(10, 1, 12, 2).astype(np.float32)
        converted_conf_data = np.random.rand(10, 1, 12).astype(np.float32)
        instance_count = np.ones(10, dtype=np.uint8)
        instance_embedding = np.zeros((10, 1, 12), dtype=np.float32)
        instance_track_id = np.zeros((10, 1), dtype=np.uint32)

        mock_convert_v2_to_v3.return_value = (
            converted_pose_data,
            converted_conf_data,
            instance_count,
            instance_embedding,
            instance_track_id,
        )

        # Act
        promote_pose_data(pose_file, current_version, new_version)

        # Assert
        # Verify HDF5 file was opened correctly
        mock_h5py_file.assert_called_once_with(pose_file, "r")

        # Verify data reshaping was done correctly
        expected_reshaped_pose = np.reshape(original_pose_data, [-1, 1, 12, 2])
        expected_reshaped_conf = np.reshape(original_conf_data, [-1, 1, 12])

        # Verify convert_v2_to_v3 was called with reshaped data
        mock_convert_v2_to_v3.assert_called_once()
        call_args = mock_convert_v2_to_v3.call_args[0]
        np.testing.assert_array_equal(call_args[0], expected_reshaped_pose)
        np.testing.assert_array_equal(call_args[1], expected_reshaped_conf)

        # Verify write functions were called
        mock_write_pose_v2_data.assert_called_once_with(
            pose_file,
            converted_pose_data,
            converted_conf_data,
            "test_config",
            "test_model",
        )
        mock_write_pose_v3_data.assert_called_once_with(
            pose_file, instance_count, instance_embedding, instance_track_id
        )

    @patch("mouse_tracking.utils.writers.write_pose_v3_data")
    @patch("mouse_tracking.utils.writers.write_pose_v2_data")
    @patch("mouse_tracking.utils.writers.convert_v2_to_v3")
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_v2_to_v3_missing_attributes(
        self,
        mock_h5py_file,
        mock_convert_v2_to_v3,
        mock_write_pose_v2_data,
        mock_write_pose_v3_data,
    ):
        """Test v2 to v3 promotion when config/model attributes are missing."""
        # Arrange
        pose_file = "test_pose.h5"
        current_version = 2
        new_version = 3

        # Mock HDF5 file data without attributes
        mock_file_context = MagicMock()
        mock_h5py_file.return_value.__enter__.return_value = mock_file_context

        original_pose_data = np.random.rand(5, 12, 2).astype(np.float32)
        original_conf_data = np.random.rand(5, 12).astype(np.float32)

        # Mock points without attrs to raise KeyError
        mock_points = Mock(__getitem__=lambda self, slice_obj: original_pose_data)
        mock_points.attrs = {"other_attr": "value"}  # Missing 'config' and 'model'

        mock_file_context.__getitem__.side_effect = lambda key: {
            "poseest/points": mock_points,
            "poseest/confidence": Mock(
                __getitem__=lambda self, slice_obj: original_conf_data
            ),
        }[key]

        # Mock convert_v2_to_v3 return values
        mock_convert_v2_to_v3.return_value = (
            np.random.rand(5, 1, 12, 2),
            np.random.rand(5, 1, 12),
            np.ones(5, dtype=np.uint8),
            np.zeros((5, 1, 12)),
            np.zeros((5, 1)),
        )

        # Act
        promote_pose_data(pose_file, current_version, new_version)

        # Assert
        # Should use 'unknown' for missing attributes
        mock_write_pose_v2_data.assert_called_once()
        # Check that 'unknown' was passed for config and model strings
        # Use assert_called_with to verify the exact arguments
        mock_write_pose_v2_data.assert_called_with(
            pose_file,
            mock_convert_v2_to_v3.return_value[0],  # pose_data
            mock_convert_v2_to_v3.return_value[1],  # conf_data
            "unknown",  # config_str
            "unknown",  # model_str
        )

    @patch("mouse_tracking.utils.writers.write_pose_v4_data")
    @patch("mouse_tracking.utils.writers.write_pose_v3_data")
    @patch("mouse_tracking.utils.writers.write_pose_v2_data")
    @patch("mouse_tracking.utils.writers.convert_v2_to_v3")
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_v2_to_v4_skips_v3_promotion(
        self,
        mock_h5py_file,
        mock_convert_v2_to_v3,
        mock_write_pose_v2_data,
        mock_write_pose_v3_data,
        mock_write_pose_v4_data,
    ):
        """Test that v2 to v4 promotion still goes through v3 step."""
        # Arrange
        pose_file = "test_pose.h5"
        current_version = 2
        new_version = 4

        # Mock track and instance data for v3->v4 conversion
        track_data = np.array([[1], [1], [2]], dtype=np.uint32)
        instance_data = np.array([1, 1, 1], dtype=np.uint8)

        original_pose_data = np.random.rand(3, 12, 2).astype(np.float32)
        original_conf_data = np.random.rand(3, 12).astype(np.float32)

        # Setup mock to handle multiple file opening calls
        file_call_count = 0

        def mock_file_side_effect(filename, mode):
            nonlocal file_call_count
            file_call_count += 1
            mock_context = MagicMock()

            if file_call_count == 1:  # First call for v2->v3
                mock_context.__enter__.return_value.__getitem__.side_effect = (
                    lambda key: {
                        "poseest/points": Mock(
                            __getitem__=lambda self, slice_obj: original_pose_data,
                            attrs={"config": "test", "model": "test"},
                        ),
                        "poseest/confidence": Mock(
                            __getitem__=lambda self, slice_obj: original_conf_data
                        ),
                    }[key]
                )
            elif file_call_count == 2:  # Second call for v3->v4
                mock_context.__enter__.return_value.__getitem__.side_effect = (
                    lambda key: {
                        "poseest/instance_track_id": Mock(
                            __getitem__=lambda self, slice_obj: track_data
                        ),
                        "poseest/instance_count": Mock(
                            __getitem__=lambda self, slice_obj: instance_data
                        ),
                    }[key]
                )

            return mock_context

        mock_h5py_file.side_effect = mock_file_side_effect

        mock_convert_v2_to_v3.return_value = (
            np.random.rand(3, 1, 12, 2),
            np.random.rand(3, 1, 12),
            np.ones(3, dtype=np.uint8),
            np.zeros((3, 1, 12)),
            np.zeros((3, 1)),
        )

        # Act
        promote_pose_data(pose_file, current_version, new_version)

        # Assert
        # Should call v2 to v3 conversion functions and then v4 functions
        mock_convert_v2_to_v3.assert_called_once()
        mock_write_pose_v2_data.assert_called_once()
        mock_write_pose_v3_data.assert_called_once()
        mock_write_pose_v4_data.assert_called_once()


class TestPromotePoseDataV3ToV4:
    """Test v3 to v4 promotion functionality."""

    @patch("mouse_tracking.utils.writers.write_pose_v4_data")
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_v3_to_v4_single_mouse(self, mock_h5py_file, mock_write_pose_v4_data):
        """Test v3 to v4 promotion with single mouse data."""
        # Arrange
        pose_file = "test_pose.h5"
        current_version = 3
        new_version = 4

        # Mock track and instance data for single mouse
        track_data = np.array(
            [[1], [1], [2], [2], [2]], dtype=np.uint32
        )  # Two tracklets
        instance_data = np.array([1, 1, 1, 1, 1], dtype=np.uint8)  # Always 1 mouse

        mock_file_context = MagicMock()
        mock_h5py_file.return_value.__enter__.return_value = mock_file_context
        mock_file_context.__getitem__.side_effect = lambda key: {
            "poseest/instance_track_id": Mock(
                __getitem__=lambda self, slice_obj: track_data
            ),
            "poseest/instance_count": Mock(
                __getitem__=lambda self, slice_obj: instance_data
            ),
        }[key]

        # Act
        promote_pose_data(pose_file, current_version, new_version)

        # Assert
        mock_write_pose_v4_data.assert_called_once()
        call_args = mock_write_pose_v4_data.call_args[0]

        # Check that the call includes expected arguments
        assert call_args[0] == pose_file  # pose_file
        # masks should be mostly False (since single mouse case flattens tracklets)
        masks = call_args[1]
        ids = call_args[2]
        centers = call_args[3]
        embeds = call_args[4]

        # Verify shapes
        assert masks.shape == track_data.shape
        assert ids.shape == track_data.shape
        assert centers.shape == (1, 1)  # [1, num_mice] where num_mice = 1
        assert embeds.shape == (track_data.shape[0], track_data.shape[1], 1)

    @patch("mouse_tracking.utils.writers.write_pose_v4_data")
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_v3_to_v4_multi_mouse(self, mock_h5py_file, mock_write_pose_v4_data):
        """Test v3 to v4 promotion with multiple mice (longest tracks preserved)."""
        # Arrange
        pose_file = "test_pose.h5"
        current_version = 3
        new_version = 4

        # Mock track and instance data for 2 mice with varying track lengths
        track_data = np.array(
            [
                [1, 3],  # Frame 0: track 1 and 3
                [1, 3],  # Frame 1: track 1 and 3
                [1, 4],  # Frame 2: track 1 and 4
                [2, 4],  # Frame 3: track 2 and 4
                [2, 4],  # Frame 4: track 2 and 4
            ],
            dtype=np.uint32,
        )
        instance_data = np.array([2, 2, 2, 2, 2], dtype=np.uint8)  # Always 2 mice

        mock_file_context = MagicMock()
        mock_h5py_file.return_value.__enter__.return_value = mock_file_context
        mock_file_context.__getitem__.side_effect = lambda key: {
            "poseest/instance_track_id": Mock(
                __getitem__=lambda self, slice_obj: track_data
            ),
            "poseest/instance_count": Mock(
                __getitem__=lambda self, slice_obj: instance_data
            ),
        }[key]

        # Act
        promote_pose_data(pose_file, current_version, new_version)

        # Assert
        mock_write_pose_v4_data.assert_called_once()
        call_args = mock_write_pose_v4_data.call_args[0]

        masks = call_args[1]
        ids = call_args[2]
        centers = call_args[3]
        embeds = call_args[4]

        # Verify shapes for 2 mice
        assert masks.shape == track_data.shape
        assert ids.shape == track_data.shape
        assert centers.shape == (1, 2)  # [1, num_mice] where num_mice = 2
        assert embeds.shape == (track_data.shape[0], track_data.shape[1], 1)

    def test_no_promotion_if_versions_dont_match(self):
        """Test that no promotion occurs if version conditions aren't met."""
        # Arrange
        pose_file = "test_pose.h5"

        # Test cases where no promotion should occur
        test_cases = [
            (4, 4),  # same version
            (5, 4),  # current > new
            (4, 3),  # current > new
        ]

        for current_version, new_version in test_cases:
            with patch("mouse_tracking.utils.writers.h5py.File") as mock_h5py_file:
                # Act
                promote_pose_data(pose_file, current_version, new_version)

                # Assert
                # Should not open any files since no promotion needed
                mock_h5py_file.assert_not_called()


class TestPromotePoseDataV5ToV6:
    """Test v5 to v6 promotion functionality."""

    @patch("mouse_tracking.utils.writers.write_v6_tracklets")
    @patch("mouse_tracking.utils.writers.write_seg_data")
    @patch("mouse_tracking.utils.writers.hungarian_match_points_seg")
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_v5_to_v6_with_segmentation_data(
        self,
        mock_h5py_file,
        mock_hungarian_match,
        mock_write_seg_data,
        mock_write_v6_tracklets,
    ):
        """Test v5 to v6 promotion when segmentation data is present."""
        # Arrange
        pose_file = "test_pose.h5"
        current_version = 5
        new_version = 6

        # Mock pose and segmentation data
        pose_data = np.random.rand(3, 2, 12, 2).astype(np.float32)
        pose_tracks = np.array([[1, 2], [1, 2], [1, 3]], dtype=np.uint32)
        pose_ids = np.array([[10, 20], [10, 20], [10, 30]], dtype=np.uint32)
        seg_data = np.random.rand(3, 2, 1, 10, 2).astype(np.int32)

        mock_file_context = MagicMock()
        mock_h5py_file.return_value.__enter__.return_value = mock_file_context

        # Mock the 'in' operator for checking if segmentation data exists
        mock_file_context.__contains__ = lambda self, key: key == "poseest/seg_data"
        mock_file_context.__getitem__.side_effect = lambda key: {
            "poseest/points": Mock(__getitem__=lambda self, slice_obj: pose_data),
            "poseest/instance_track_id": Mock(
                __getitem__=lambda self, slice_obj: pose_tracks
            ),
            "poseest/instance_embed_id": Mock(
                __getitem__=lambda self, slice_obj: pose_ids
            ),
            "poseest/seg_data": Mock(__getitem__=lambda self, slice_obj: seg_data),
        }[key]

        # Mock Hungarian matching to return simple matches
        mock_hungarian_match.side_effect = [
            [(0, 0), (1, 1)],  # Frame 0 matches
            [(0, 0), (1, 1)],  # Frame 1 matches
            [(0, 0), (1, 1)],  # Frame 2 matches
        ]

        # Act
        promote_pose_data(pose_file, current_version, new_version)

        # Assert
        # Should call Hungarian matching for each frame
        assert mock_hungarian_match.call_count == 3

        # Should write v6 tracklets
        mock_write_v6_tracklets.assert_called_once()
        call_args = mock_write_v6_tracklets.call_args[0]

        seg_tracks = call_args[1]
        seg_ids = call_args[2]

        # Verify shapes
        assert seg_tracks.shape == seg_data.shape[:2]
        assert seg_ids.shape == seg_data.shape[:2]

        # Should not write seg_data since it already exists
        mock_write_seg_data.assert_not_called()

    @patch("mouse_tracking.utils.writers.write_v6_tracklets")
    @patch("mouse_tracking.utils.writers.write_seg_data")
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_v5_to_v6_without_segmentation_data(
        self, mock_h5py_file, mock_write_seg_data, mock_write_v6_tracklets
    ):
        """Test v5 to v6 promotion when segmentation data is missing."""
        # Arrange
        pose_file = "test_pose.h5"
        current_version = 5
        new_version = 6

        # Mock pose data without segmentation
        pose_shape = (4, 2, 12, 2)

        mock_file_context = MagicMock()
        mock_h5py_file.return_value.__enter__.return_value = mock_file_context

        # Mock that segmentation data is NOT present
        mock_file_context.__contains__ = lambda self, key: key != "poseest/seg_data"

        # Create a mock with shape attribute
        mock_points = Mock()
        mock_points.shape = pose_shape
        mock_file_context.__getitem__.side_effect = lambda key: {
            "poseest/points": mock_points,
        }[key]

        # Act
        promote_pose_data(pose_file, current_version, new_version)

        # Assert
        # Should write default segmentation data
        mock_write_seg_data.assert_called_once()
        call_args = mock_write_seg_data.call_args

        # Check that default seg_data was created with correct shape
        seg_data = call_args[0][1]
        expected_shape = (pose_shape[0], 1, 1, 1, 2)
        assert seg_data.shape == expected_shape
        assert np.all(seg_data == -1)  # Should be filled with -1

        # Should write v6 tracklets with default values
        mock_write_v6_tracklets.assert_called_once()


class TestPromotePoseDataEdgeCases:
    """Test edge cases and error conditions."""

    def test_no_promotion_needed_same_version(self):
        """Test that no work is done when current_version == new_version."""
        # Arrange
        pose_file = "test_pose.h5"
        current_version = 3
        new_version = 3

        with patch("mouse_tracking.utils.writers.h5py.File") as mock_h5py_file:
            # Act
            promote_pose_data(pose_file, current_version, new_version)

            # Assert
            mock_h5py_file.assert_not_called()

    def test_no_promotion_current_higher_than_new(self):
        """Test that no work is done when current_version > new_version."""
        # Arrange
        pose_file = "test_pose.h5"
        current_version = 5
        new_version = 3

        with patch("mouse_tracking.utils.writers.h5py.File") as mock_h5py_file:
            # Act
            promote_pose_data(pose_file, current_version, new_version)

            # Assert
            mock_h5py_file.assert_not_called()

    @pytest.mark.parametrize(
        "current_version,new_version,expected_v2_to_v3,expected_v3_to_v4,expected_v5_to_v6",
        [
            (2, 3, True, False, False),  # Only v2 to v3
            (2, 4, True, True, False),  # v2 to v3, then v3 to v4
            (2, 6, True, True, True),  # All promotions
            (3, 4, False, True, False),  # Only v3 to v4
            (3, 6, False, True, True),  # v3 to v4, then v5 to v6
            (4, 6, False, False, True),  # Only v5 to v6 (note: v4->v5 is no-op)
            (5, 6, False, False, True),  # Only v5 to v6
        ],
        ids=[
            "v2_to_v3_only",
            "v2_to_v4",
            "v2_to_v6_full",
            "v3_to_v4_only",
            "v3_to_v6",
            "v4_to_v6",
            "v5_to_v6_only",
        ],
    )
    def test_version_promotion_paths(
        self,
        current_version,
        new_version,
        expected_v2_to_v3,
        expected_v3_to_v4,
        expected_v5_to_v6,
    ):
        """Test that correct promotion paths are taken for different version combinations."""
        pose_file = "test_pose.h5"

        # Create mock data
        original_pose_data = np.random.rand(3, 12, 2).astype(np.float32)
        original_conf_data = np.random.rand(3, 12).astype(np.float32)
        track_data = np.array([[1], [1], [2]], dtype=np.uint32)
        instance_data = np.array([1, 1, 1], dtype=np.uint8)
        pose_shape = (3, 1, 12, 2)

        def mock_file_side_effect(filename, mode):
            mock_context = MagicMock()
            mock_file_context = MagicMock()

            # Create mocks that work for all version transitions
            mock_points = Mock(
                __getitem__=lambda self, slice_obj: original_pose_data,
                attrs={"config": "test", "model": "test"},
            )
            mock_points.shape = pose_shape

            mock_file_context.__getitem__.side_effect = lambda key: {
                "poseest/points": mock_points,
                "poseest/confidence": Mock(
                    __getitem__=lambda self, slice_obj: original_conf_data
                ),
                "poseest/instance_track_id": Mock(
                    __getitem__=lambda self, slice_obj: track_data
                ),
                "poseest/instance_count": Mock(
                    __getitem__=lambda self, slice_obj: instance_data
                ),
                "poseest/instance_embed_id": Mock(
                    __getitem__=lambda self, slice_obj: track_data
                ),
            }.get(key, Mock())

            mock_file_context.__contains__ = lambda self, key: key != "poseest/seg_data"
            mock_context.__enter__.return_value = mock_file_context
            return mock_context

        with (
            patch(
                "mouse_tracking.utils.writers.h5py.File",
                side_effect=mock_file_side_effect,
            ),
            patch(
                "mouse_tracking.utils.writers.convert_v2_to_v3",
                return_value=(
                    np.random.rand(3, 1, 12, 2),
                    np.random.rand(3, 1, 12),
                    np.ones(3, dtype=np.uint8),
                    np.zeros((3, 1, 12)),
                    np.zeros((3, 1)),
                ),
            ),
            patch("mouse_tracking.utils.writers.write_pose_v2_data"),
            patch("mouse_tracking.utils.writers.write_pose_v3_data"),
            patch("mouse_tracking.utils.writers.write_pose_v4_data"),
            patch("mouse_tracking.utils.writers.write_v6_tracklets"),
            patch("mouse_tracking.utils.writers.write_seg_data"),
            patch(
                "mouse_tracking.utils.writers.hungarian_match_points_seg",
                return_value=[(0, 0)],
            ),
        ):
            # The function should handle the version transitions correctly
            promote_pose_data(pose_file, current_version, new_version)


class TestPromotePoseDataIntegration:
    """Integration-style tests that exercise multiple components together."""

    @patch("mouse_tracking.utils.writers.hungarian_match_points_seg")
    @patch("mouse_tracking.utils.writers.write_v6_tracklets")
    @patch("mouse_tracking.utils.writers.write_seg_data")
    @patch("mouse_tracking.utils.writers.write_pose_v4_data")
    @patch("mouse_tracking.utils.writers.write_pose_v3_data")
    @patch("mouse_tracking.utils.writers.write_pose_v2_data")
    @patch("mouse_tracking.utils.writers.convert_v2_to_v3")
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_full_v2_to_v6_promotion(
        self,
        mock_h5py_file,
        mock_convert_v2_to_v3,
        mock_write_pose_v2_data,
        mock_write_pose_v3_data,
        mock_write_pose_v4_data,
        mock_write_seg_data,
        mock_write_v6_tracklets,
        mock_hungarian_match,
    ):
        """Test complete promotion from v2 to v6."""
        # Arrange
        pose_file = "test_pose.h5"
        current_version = 2
        new_version = 6

        # Setup complex mock that handles multiple file opening contexts
        original_pose_data = np.random.rand(5, 12, 2).astype(np.float32)
        original_conf_data = np.random.rand(5, 12).astype(np.float32)

        # Setup mock data for different file reads
        track_data = np.array([[1], [1], [2], [2], [2]], dtype=np.uint32)
        instance_data = np.array([1, 1, 1, 1, 1], dtype=np.uint8)
        pose_shape = (5, 1, 12, 2)

        def mock_file_side_effect(filename, mode):
            mock_context = MagicMock()
            mock_file_context = MagicMock()

            # Setup data for all possible reads during promotion
            mock_file_context.__getitem__.side_effect = lambda key: {
                "poseest/points": Mock(
                    __getitem__=lambda self, slice_obj: original_pose_data,
                    attrs={"config": "test", "model": "test"},
                    shape=pose_shape,
                ),
                "poseest/confidence": Mock(
                    __getitem__=lambda self, slice_obj: original_conf_data
                ),
                "poseest/instance_track_id": Mock(
                    __getitem__=lambda self, slice_obj: track_data
                ),
                "poseest/instance_count": Mock(
                    __getitem__=lambda self, slice_obj: instance_data
                ),
            }.get(key, Mock())

            mock_file_context.__contains__ = lambda self, key: key != "poseest/seg_data"
            mock_context.__enter__.return_value = mock_file_context
            return mock_context

        mock_h5py_file.side_effect = mock_file_side_effect

        # Mock convert function
        mock_convert_v2_to_v3.return_value = (
            np.random.rand(5, 1, 12, 2),
            np.random.rand(5, 1, 12),
            np.ones(5, dtype=np.uint8),
            np.zeros((5, 1, 12)),
            np.zeros((5, 1)),
        )

        # Mock hungarian matching
        mock_hungarian_match.return_value = [(0, 0)]

        # Act
        promote_pose_data(pose_file, current_version, new_version)

        # Assert
        # Should call all the write functions in sequence
        mock_write_pose_v2_data.assert_called_once()
        mock_write_pose_v3_data.assert_called_once()
        mock_write_pose_v4_data.assert_called_once()
        mock_write_seg_data.assert_called_once()
        mock_write_v6_tracklets.assert_called_once()
