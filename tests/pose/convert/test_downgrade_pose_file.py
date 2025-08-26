"""
Unit tests for downgrade_pose_file function.

Tests cover file I/O operations, version handling, error conditions,
and successful downgrade scenarios with proper mocking of HDF5 operations.
"""

from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from mouse_tracking.core.exceptions import InvalidPoseFileException
from mouse_tracking.pose.convert import downgrade_pose_file


def _create_mock_h5_file_context(data_dict, attrs_dict):
    """Helper function to create a mock H5 file context manager.

    Args:
        data_dict: Dictionary of dataset paths to numpy arrays
        attrs_dict: Dictionary of attribute paths to attribute dictionaries

    Returns:
        Mock object that can be used as H5 file context manager
    """
    mock_file = MagicMock()

    def mock_getitem(key):
        if key in data_dict:
            mock_dataset = MagicMock()
            mock_dataset.__getitem__.return_value = data_dict[key]
            if key in attrs_dict:
                mock_dataset.attrs = attrs_dict[key]
            else:
                mock_dataset.attrs = {}
            return mock_dataset
        elif key in attrs_dict:
            mock_group = MagicMock()
            mock_group.attrs = attrs_dict[key]
            return mock_group
        else:
            raise KeyError(f"Mock key {key} not found")

    mock_file.__enter__.return_value = mock_file
    mock_file.__exit__.return_value = None
    mock_file.__getitem__.side_effect = mock_getitem

    return mock_file


class TestDowngradePoseFileErrorHandling:
    """Test error handling scenarios for downgrade_pose_file."""

    def test_missing_file_raises_file_not_found_error(self):
        """Test that missing input file raises FileNotFoundError."""
        with (
            patch("mouse_tracking.pose.convert.os.path.isfile", return_value=False),
            pytest.raises(
                FileNotFoundError, match="ERROR: missing file: nonexistent.h5"
            ),
        ):
            downgrade_pose_file("nonexistent.h5")

    def test_missing_version_attribute_raises_invalid_pose_file_exception(self):
        """Test that files without version attribute raise InvalidPoseFileException."""
        mock_h5 = _create_mock_h5_file_context(
            data_dict={},
            attrs_dict={"poseest": {}},  # No version attribute
        )

        with (
            patch("mouse_tracking.pose.convert.os.path.isfile", return_value=True),
            patch("mouse_tracking.pose.convert.h5py.File", return_value=mock_h5),
            pytest.raises(
                InvalidPoseFileException,
                match="Pose file test.h5 did not have a valid version",
            ),
        ):
            downgrade_pose_file("test.h5")

    @patch("mouse_tracking.pose.convert.exit")
    def test_v2_file_prints_message_and_exits(self, mock_exit):
        """Test that v2 files print message and exit gracefully."""
        # Make exit raise SystemExit to actually terminate execution
        mock_exit.side_effect = SystemExit(0)

        # For v2 files, we just need version info since function exits early
        mock_h5 = _create_mock_h5_file_context(
            data_dict={}, attrs_dict={"poseest": {"version": [2]}}
        )

        with (
            patch("mouse_tracking.pose.convert.os.path.isfile", return_value=True),
            patch("mouse_tracking.pose.convert.h5py.File", return_value=mock_h5),
            patch("builtins.print") as mock_print,
        ):
            with pytest.raises(SystemExit) as exc_info:
                downgrade_pose_file("test_v2.h5")

            assert exc_info.value.code == 0
            mock_print.assert_called_once_with(
                "Pose file test_v2.h5 is already v2. Exiting."
            )
            mock_exit.assert_called_once_with(0)


class TestDowngradePoseFileV3Processing:
    """Test successful processing of v3 pose files."""

    @patch("mouse_tracking.pose.convert.write_pixel_per_cm_attr")
    @patch("mouse_tracking.pose.convert.write_pose_v2_data")
    @patch("mouse_tracking.pose.convert.multi_to_v2")
    def test_v3_file_basic_processing(
        self, mock_multi_to_v2, mock_write_v2, mock_write_pixel
    ):
        """Test basic v3 file processing with minimal data."""
        # Create test data
        pose_data = np.random.rand(10, 2, 12, 2).astype(np.float32)
        conf_data = np.random.rand(10, 2, 12).astype(np.float32)
        track_id = np.array([[1, 0], [1, 2], [0, 2], [1, 2]], dtype=np.uint32)

        mock_h5 = _create_mock_h5_file_context(
            data_dict={
                "poseest/points": pose_data,
                "poseest/confidence": conf_data,
                "poseest/instance_track_id": track_id,
            },
            attrs_dict={
                "poseest": {"version": [3]},
                "poseest/points": {"config": "test_config", "model": "test_model"},
            },
        )

        # Mock multi_to_v2 return value
        mock_multi_to_v2.return_value = [
            (1, np.random.rand(10, 12, 2), np.random.rand(10, 12)),
            (2, np.random.rand(10, 12, 2), np.random.rand(10, 12)),
        ]

        with (
            patch("mouse_tracking.pose.convert.os.path.isfile", return_value=True),
            patch("mouse_tracking.pose.convert.h5py.File", return_value=mock_h5),
        ):
            downgrade_pose_file("test_pose_est_v3.h5")

            # Verify multi_to_v2 was called with correct arguments
            mock_multi_to_v2.assert_called_once()
            args = mock_multi_to_v2.call_args[0]
            np.testing.assert_array_equal(args[0], pose_data)
            np.testing.assert_array_equal(args[1], conf_data)
            np.testing.assert_array_equal(args[2], track_id)

            # Verify output files were written
            expected_calls = [
                call(
                    "test_animal_1_pose_est_v2.h5",
                    mock_multi_to_v2.return_value[0][1],
                    mock_multi_to_v2.return_value[0][2],
                    "test_config",
                    "test_model",
                ),
                call(
                    "test_animal_2_pose_est_v2.h5",
                    mock_multi_to_v2.return_value[1][1],
                    mock_multi_to_v2.return_value[1][2],
                    "test_config",
                    "test_model",
                ),
            ]
            mock_write_v2.assert_has_calls(expected_calls)

            # Verify pixel scaling was not written (no pixel data)
            mock_write_pixel.assert_not_called()

    @patch("mouse_tracking.pose.convert.write_pixel_per_cm_attr")
    @patch("mouse_tracking.pose.convert.write_pose_v2_data")
    @patch("mouse_tracking.pose.convert.multi_to_v2")
    def test_v3_file_with_pixel_scaling(
        self, mock_multi_to_v2, mock_write_v2, mock_write_pixel
    ):
        """Test v3 file processing with pixel scaling attributes."""
        pose_data = np.random.rand(5, 1, 12, 2).astype(np.float32)
        conf_data = np.random.rand(5, 1, 12).astype(np.float32)
        track_id = np.ones((5, 1), dtype=np.uint32)

        mock_h5 = _create_mock_h5_file_context(
            data_dict={
                "poseest/points": pose_data,
                "poseest/confidence": conf_data,
                "poseest/instance_track_id": track_id,
            },
            attrs_dict={
                "poseest": {
                    "version": [3],
                    "cm_per_pixel": 0.1,
                    "cm_per_pixel_source": "manual",
                },
                "poseest/points": {"config": "test_config", "model": "test_model"},
            },
        )

        mock_multi_to_v2.return_value = [
            (1, np.random.rand(5, 12, 2), np.random.rand(5, 12))
        ]

        with (
            patch("mouse_tracking.pose.convert.os.path.isfile", return_value=True),
            patch("mouse_tracking.pose.convert.h5py.File", return_value=mock_h5),
        ):
            downgrade_pose_file("experiment_pose_est_v3.h5")

            # Verify pixel scaling was written
            mock_write_pixel.assert_called_once_with(
                "experiment_animal_1_pose_est_v2.h5", 0.1, "manual"
            )

    @patch("mouse_tracking.pose.convert.write_pose_v2_data")
    @patch("mouse_tracking.pose.convert.multi_to_v2")
    def test_v3_file_missing_config_model_attributes(
        self, mock_multi_to_v2, mock_write_v2
    ):
        """Test v3 file processing when config/model attributes are missing."""
        pose_data = np.random.rand(3, 1, 12, 2).astype(np.float32)
        conf_data = np.random.rand(3, 1, 12).astype(np.float32)
        track_id = np.ones((3, 1), dtype=np.uint32)

        mock_h5 = _create_mock_h5_file_context(
            data_dict={
                "poseest/points": pose_data,
                "poseest/confidence": conf_data,
                "poseest/instance_track_id": track_id,
            },
            attrs_dict={
                "poseest": {"version": [3]},
                "poseest/points": {},  # Missing config and model
            },
        )

        mock_multi_to_v2.return_value = [
            (1, np.random.rand(3, 12, 2), np.random.rand(3, 12))
        ]

        with (
            patch("mouse_tracking.pose.convert.os.path.isfile", return_value=True),
            patch("mouse_tracking.pose.convert.h5py.File", return_value=mock_h5),
        ):
            downgrade_pose_file("test_pose_est_v3.h5")

            # Verify 'unknown' is used for missing config/model
            mock_write_v2.assert_called_once_with(
                "test_animal_1_pose_est_v2.h5",
                mock_multi_to_v2.return_value[0][1],
                mock_multi_to_v2.return_value[0][2],
                "unknown",
                "unknown",
            )


class TestDowngradePoseFileV4Processing:
    """Test successful processing of v4+ pose files."""

    @patch("mouse_tracking.pose.convert.write_pose_v2_data")
    @patch("mouse_tracking.pose.convert.multi_to_v2")
    def test_v4_file_uses_embed_id_by_default(self, mock_multi_to_v2, mock_write_v2):
        """Test that v4+ files use instance_embed_id by default."""
        pose_data = np.random.rand(8, 3, 12, 2).astype(np.float32)
        conf_data = np.random.rand(8, 3, 12).astype(np.float32)
        embed_id = np.array([[1, 2, 0], [1, 0, 3], [2, 3, 0]], dtype=np.uint32)
        track_id = np.array([[10, 20, 0], [10, 0, 30], [20, 30, 0]], dtype=np.uint32)

        mock_h5 = _create_mock_h5_file_context(
            data_dict={
                "poseest/points": pose_data,
                "poseest/confidence": conf_data,
                "poseest/instance_embed_id": embed_id,
                "poseest/instance_track_id": track_id,
            },
            attrs_dict={
                "poseest": {"version": [4]},
                "poseest/points": {"config": "v4_config", "model": "v4_model"},
            },
        )

        mock_multi_to_v2.return_value = [
            (1, np.random.rand(8, 12, 2), np.random.rand(8, 12)),
            (2, np.random.rand(8, 12, 2), np.random.rand(8, 12)),
            (3, np.random.rand(8, 12, 2), np.random.rand(8, 12)),
        ]

        with (
            patch("mouse_tracking.pose.convert.os.path.isfile", return_value=True),
            patch("mouse_tracking.pose.convert.h5py.File", return_value=mock_h5),
        ):
            downgrade_pose_file("data_pose_est_v4.h5")

            # Verify multi_to_v2 was called with embed_id (not track_id)
            args = mock_multi_to_v2.call_args[0]
            np.testing.assert_array_equal(args[2], embed_id)

    @patch("mouse_tracking.pose.convert.write_pose_v2_data")
    @patch("mouse_tracking.pose.convert.multi_to_v2")
    def test_v4_file_uses_track_id_when_disabled(self, mock_multi_to_v2, mock_write_v2):
        """Test that v4+ files use instance_track_id when disable_id=True."""
        pose_data = np.random.rand(5, 2, 12, 2).astype(np.float32)
        conf_data = np.random.rand(5, 2, 12).astype(np.float32)
        embed_id = np.array([[1, 2], [1, 0]], dtype=np.uint32)
        track_id = np.array([[10, 20], [10, 0]], dtype=np.uint32)

        mock_h5 = _create_mock_h5_file_context(
            data_dict={
                "poseest/points": pose_data,
                "poseest/confidence": conf_data,
                "poseest/instance_embed_id": embed_id,
                "poseest/instance_track_id": track_id,
            },
            attrs_dict={
                "poseest": {"version": [5]},
                "poseest/points": {"config": "v5_config", "model": "v5_model"},
            },
        )

        mock_multi_to_v2.return_value = [
            (10, np.random.rand(5, 12, 2), np.random.rand(5, 12)),
            (20, np.random.rand(5, 12, 2), np.random.rand(5, 12)),
        ]

        with (
            patch("mouse_tracking.pose.convert.os.path.isfile", return_value=True),
            patch("mouse_tracking.pose.convert.h5py.File", return_value=mock_h5),
        ):
            downgrade_pose_file("data_pose_est_v5.h5", disable_id=True)

            # Verify multi_to_v2 was called with track_id (not embed_id)
            args = mock_multi_to_v2.call_args[0]
            np.testing.assert_array_equal(args[2], track_id)


class TestDowngradePoseFileFilenameHandling:
    """Test filename pattern replacement functionality."""

    @patch("mouse_tracking.pose.convert.write_pose_v2_data")
    @patch("mouse_tracking.pose.convert.multi_to_v2")
    def test_various_filename_patterns(self, mock_multi_to_v2, mock_write_v2):
        """Test that different version filename patterns are handled correctly."""
        test_cases = [
            ("experiment_pose_est_v3.h5", "experiment_animal_1_pose_est_v2.h5"),
            ("data_pose_est_v10.h5", "data_animal_1_pose_est_v2.h5"),
            ("mouse_pose_est_v6.h5", "mouse_animal_1_pose_est_v2.h5"),
            (
                "test.h5",
                "test.h5_animal_1_pose_est_v2.h5",
            ),  # No version pattern to replace
        ]

        for input_file, expected_output in test_cases:
            with (
                self._setup_basic_v3_mock(mock_multi_to_v2),
                patch("mouse_tracking.pose.convert.os.path.isfile", return_value=True),
                patch(
                    "mouse_tracking.pose.convert.h5py.File",
                    return_value=self.mock_h5,
                ),
            ):
                downgrade_pose_file(input_file)

                # Check that the correct output filename was used
                mock_write_v2.assert_called_once()
                actual_output = mock_write_v2.call_args[0][0]
                assert actual_output == expected_output, (
                    f"Expected {expected_output}, got {actual_output}"
                )

                mock_write_v2.reset_mock()

    def _setup_basic_v3_mock(self, mock_multi_to_v2):
        """Helper to set up basic v3 file mock."""
        pose_data = np.random.rand(2, 1, 12, 2).astype(np.float32)
        conf_data = np.random.rand(2, 1, 12).astype(np.float32)
        track_id = np.ones((2, 1), dtype=np.uint32)

        self.mock_h5 = _create_mock_h5_file_context(
            data_dict={
                "poseest/points": pose_data,
                "poseest/confidence": conf_data,
                "poseest/instance_track_id": track_id,
            },
            attrs_dict={
                "poseest": {"version": [3]},
                "poseest/points": {"config": "test", "model": "test"},
            },
        )

        mock_multi_to_v2.return_value = [
            (1, np.random.rand(2, 12, 2), np.random.rand(2, 12))
        ]

        return self.mock_h5


class TestDowngradePoseFileEdgeCases:
    """Test edge cases and unusual scenarios."""

    @patch("mouse_tracking.pose.convert.write_pose_v2_data")
    @patch("mouse_tracking.pose.convert.multi_to_v2")
    def test_empty_multi_to_v2_result(self, mock_multi_to_v2, mock_write_v2):
        """Test behavior when multi_to_v2 returns no animals."""
        pose_data = np.zeros((5, 2, 12, 2), dtype=np.float32)
        conf_data = np.zeros((5, 2, 12), dtype=np.float32)
        track_id = np.zeros((5, 2), dtype=np.uint32)

        mock_h5 = _create_mock_h5_file_context(
            data_dict={
                "poseest/points": pose_data,
                "poseest/confidence": conf_data,
                "poseest/instance_track_id": track_id,
            },
            attrs_dict={
                "poseest": {"version": [3]},
                "poseest/points": {"config": "test", "model": "test"},
            },
        )

        mock_multi_to_v2.return_value = []  # No animals found

        with (
            patch("mouse_tracking.pose.convert.os.path.isfile", return_value=True),
            patch("mouse_tracking.pose.convert.h5py.File", return_value=mock_h5),
        ):
            downgrade_pose_file("empty_pose_est_v3.h5")

            # Verify no files were written
            mock_write_v2.assert_not_called()

    @patch("mouse_tracking.pose.convert.write_pose_v2_data")
    @patch("mouse_tracking.pose.convert.multi_to_v2")
    def test_single_animal_result(self, mock_multi_to_v2, mock_write_v2):
        """Test processing with only one animal in the data."""
        pose_data = np.random.rand(10, 1, 12, 2).astype(np.float32)
        conf_data = np.random.rand(10, 1, 12).astype(np.float32)
        track_id = np.ones((10, 1), dtype=np.uint32) * 5

        mock_h5 = _create_mock_h5_file_context(
            data_dict={
                "poseest/points": pose_data,
                "poseest/confidence": conf_data,
                "poseest/instance_track_id": track_id,
            },
            attrs_dict={
                "poseest": {"version": [3]},
                "poseest/points": {"config": "single_config", "model": "single_model"},
            },
        )

        mock_multi_to_v2.return_value = [
            (5, np.random.rand(10, 12, 2), np.random.rand(10, 12))
        ]

        with (
            patch("mouse_tracking.pose.convert.os.path.isfile", return_value=True),
            patch("mouse_tracking.pose.convert.h5py.File", return_value=mock_h5),
        ):
            downgrade_pose_file("single_pose_est_v3.h5")

            # Verify only one file was written with ID 5
            mock_write_v2.assert_called_once_with(
                "single_animal_5_pose_est_v2.h5",
                mock_multi_to_v2.return_value[0][1],
                mock_multi_to_v2.return_value[0][2],
                "single_config",
                "single_model",
            )

    @patch("mouse_tracking.pose.convert.write_pose_v2_data")
    @patch("mouse_tracking.pose.convert.multi_to_v2")
    def test_large_animal_ids(self, mock_multi_to_v2, mock_write_v2):
        """Test processing with large animal ID numbers."""
        pose_data = np.random.rand(3, 2, 12, 2).astype(np.float32)
        conf_data = np.random.rand(3, 2, 12).astype(np.float32)
        track_id = np.array([[1000, 0], [1000, 9999], [0, 9999]], dtype=np.uint32)

        mock_h5 = _create_mock_h5_file_context(
            data_dict={
                "poseest/points": pose_data,
                "poseest/confidence": conf_data,
                "poseest/instance_track_id": track_id,
            },
            attrs_dict={
                "poseest": {"version": [3]},
                "poseest/points": {"config": "large_config", "model": "large_model"},
            },
        )

        mock_multi_to_v2.return_value = [
            (1000, np.random.rand(3, 12, 2), np.random.rand(3, 12)),
            (9999, np.random.rand(3, 12, 2), np.random.rand(3, 12)),
        ]

        with (
            patch("mouse_tracking.pose.convert.os.path.isfile", return_value=True),
            patch("mouse_tracking.pose.convert.h5py.File", return_value=mock_h5),
        ):
            downgrade_pose_file("large_ids_pose_est_v3.h5")

            # Verify both large ID files were written
            expected_calls = [
                call(
                    "large_ids_animal_1000_pose_est_v2.h5",
                    mock_multi_to_v2.return_value[0][1],
                    mock_multi_to_v2.return_value[0][2],
                    "large_config",
                    "large_model",
                ),
                call(
                    "large_ids_animal_9999_pose_est_v2.h5",
                    mock_multi_to_v2.return_value[1][1],
                    mock_multi_to_v2.return_value[1][2],
                    "large_config",
                    "large_model",
                ),
            ]
            mock_write_v2.assert_has_calls(expected_calls, any_order=True)


class TestDowngradePoseFileIntegration:
    """Test integration scenarios that combine multiple aspects."""

    @patch("mouse_tracking.pose.convert.write_pixel_per_cm_attr")
    @patch("mouse_tracking.pose.convert.write_pose_v2_data")
    @patch("mouse_tracking.pose.convert.multi_to_v2")
    def test_realistic_multi_animal_v4_scenario(
        self, mock_multi_to_v2, mock_write_v2, mock_write_pixel
    ):
        """Test realistic scenario with multiple animals, pixel scaling, and v4 data."""
        # Create realistic multi-animal data
        pose_data = (
            np.random.rand(100, 3, 12, 2).astype(np.float32) * 500
        )  # Realistic pixel coords
        conf_data = np.random.rand(100, 3, 12).astype(np.float32)
        embed_id = np.random.choice([0, 1, 2, 3], size=(100, 3), p=[0.4, 0.2, 0.2, 0.2])

        mock_h5 = _create_mock_h5_file_context(
            data_dict={
                "poseest/points": pose_data,
                "poseest/confidence": conf_data,
                "poseest/instance_embed_id": embed_id,
                "poseest/instance_track_id": np.random.randint(0, 50, size=(100, 3)),
            },
            attrs_dict={
                "poseest": {
                    "version": [4],
                    "cm_per_pixel": 0.08,
                    "cm_per_pixel_source": "automated_calibration",
                },
                "poseest/points": {
                    "config": "production_config_v2.yaml",
                    "model": "multi_mouse_hrnet_w32_256x256_epoch_200",
                },
            },
        )

        # Mock realistic multi_to_v2 output
        mock_multi_to_v2.return_value = [
            (1, np.random.rand(100, 12, 2), np.random.rand(100, 12)),
            (2, np.random.rand(100, 12, 2), np.random.rand(100, 12)),
            (3, np.random.rand(100, 12, 2), np.random.rand(100, 12)),
        ]

        with (
            patch("mouse_tracking.pose.convert.os.path.isfile", return_value=True),
            patch("mouse_tracking.pose.convert.h5py.File", return_value=mock_h5),
        ):
            downgrade_pose_file("experiment_20241201_cage1_pose_est_v4.h5")

            # Verify all animals were processed
            assert mock_write_v2.call_count == 3

            # Verify pixel scaling was applied to all files
            expected_pixel_calls = [
                call(
                    "experiment_20241201_cage1_animal_1_pose_est_v2.h5",
                    0.08,
                    "automated_calibration",
                ),
                call(
                    "experiment_20241201_cage1_animal_2_pose_est_v2.h5",
                    0.08,
                    "automated_calibration",
                ),
                call(
                    "experiment_20241201_cage1_animal_3_pose_est_v2.h5",
                    0.08,
                    "automated_calibration",
                ),
            ]
            mock_write_pixel.assert_has_calls(expected_pixel_calls, any_order=True)

            # Verify embed_id was used (not track_id)
            args = mock_multi_to_v2.call_args[0]
            np.testing.assert_array_equal(args[2], embed_id)

    @patch("mouse_tracking.pose.convert.write_pose_v2_data")
    @patch("mouse_tracking.pose.convert.multi_to_v2")
    def test_v6_file_with_missing_optional_attributes(
        self, mock_multi_to_v2, mock_write_v2
    ):
        """Test processing v6 file with some missing optional attributes."""
        pose_data = np.ones((20, 4, 12, 2), dtype=np.float32)  # Use fixed data
        conf_data = np.ones((20, 4, 12), dtype=np.float32)
        embed_id = np.ones((20, 4), dtype=np.uint32)

        # Mock file with only some attributes present
        mock_h5 = _create_mock_h5_file_context(
            data_dict={
                "poseest/points": pose_data,
                "poseest/confidence": conf_data,
                "poseest/instance_embed_id": embed_id,
                "poseest/instance_track_id": np.ones((20, 4), dtype=np.uint32),
            },
            attrs_dict={
                "poseest": {
                    "version": [6],
                    "cm_per_pixel_source": "manual",  # Missing cm_per_pixel value
                },
                "poseest/points": {
                    "config": "v6_config",
                    "model": "v6_model",  # Both present, but missing cm_per_pixel value above
                },
            },
        )

        # Use fixed return data to make assertions predictable
        fixed_pose_1 = np.ones((20, 12, 2), dtype=np.float32)
        fixed_conf_1 = np.ones((20, 12), dtype=np.float32)
        fixed_pose_2 = np.ones((20, 12, 2), dtype=np.float32) * 2
        fixed_conf_2 = np.ones((20, 12), dtype=np.float32) * 2

        mock_multi_to_v2.return_value = [
            (1, fixed_pose_1, fixed_conf_1),
            (2, fixed_pose_2, fixed_conf_2),
        ]

        with (
            patch("mouse_tracking.pose.convert.os.path.isfile", return_value=True),
            patch("mouse_tracking.pose.convert.h5py.File", return_value=mock_h5),
        ):
            downgrade_pose_file("advanced_pose_est_v6.h5")

            # Verify files were written with config and model preserved, missing pixel scaling
            expected_calls = [
                call(
                    "advanced_animal_1_pose_est_v2.h5",
                    fixed_pose_1,
                    fixed_conf_1,
                    "v6_config",
                    "v6_model",
                ),
                call(
                    "advanced_animal_2_pose_est_v2.h5",
                    fixed_pose_2,
                    fixed_conf_2,
                    "v6_config",
                    "v6_model",
                ),
            ]
            mock_write_v2.assert_has_calls(expected_calls, any_order=True)
