"""Comprehensive unit tests for the write_v6_tracklets function."""

from unittest.mock import patch

import numpy as np
import pytest

from mouse_tracking.core.exceptions import InvalidPoseFileException
from mouse_tracking.utils.writers import write_v6_tracklets

from .mock_hdf5 import create_mock_h5_context


class TestWriteV6TrackletsBasicFunctionality:
    """Test basic functionality of write_v6_tracklets."""

    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_write_v6_tracklets_success(self, mock_h5py_file):
        """Test successful writing of v6 tracklet data."""
        # Arrange
        pose_file = "test_pose.h5"
        seg_data_shape = (100, 3, 5, 10, 2)  # [frame, num_animals, ...]
        segmentation_tracks = np.random.randint(0, 10, size=(100, 3), dtype=np.uint32)
        segmentation_ids = np.random.randint(0, 5, size=(100, 3), dtype=np.uint32)

        existing_datasets = ["poseest/seg_data"]
        mock_context = create_mock_h5_context(
            existing_datasets, seg_data_shape=seg_data_shape
        )
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Act
        write_v6_tracklets(pose_file, segmentation_tracks, segmentation_ids)

        # Assert
        # Should open file in append mode
        mock_h5py_file.assert_called_once_with(pose_file, "a")

        # Should create instance_seg_id dataset
        assert "poseest/instance_seg_id" in mock_context.created_datasets
        instance_seg_info = mock_context.created_datasets["poseest/instance_seg_id"]
        np.testing.assert_array_equal(
            instance_seg_info["data"], segmentation_tracks.astype(np.uint32)
        )

        # Should create longterm_seg_id dataset
        assert "poseest/longterm_seg_id" in mock_context.created_datasets
        longterm_seg_info = mock_context.created_datasets["poseest/longterm_seg_id"]
        np.testing.assert_array_equal(
            longterm_seg_info["data"], segmentation_ids.astype(np.uint32)
        )

    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_write_v6_tracklets_overwrite_existing(self, mock_h5py_file):
        """Test that existing tracklet datasets are properly overwritten."""
        # Arrange
        pose_file = "test_pose.h5"
        seg_data_shape = (50, 2, 1, 8, 2)
        segmentation_tracks = np.random.randint(1, 3, size=(50, 2), dtype=np.uint32)
        segmentation_ids = np.random.randint(1, 4, size=(50, 2), dtype=np.uint32)

        existing_datasets = [
            "poseest/seg_data",
            "poseest/instance_seg_id",
            "poseest/longterm_seg_id",
        ]
        mock_context = create_mock_h5_context(
            existing_datasets, seg_data_shape=seg_data_shape
        )
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Act
        write_v6_tracklets(pose_file, segmentation_tracks, segmentation_ids)

        # Assert
        # Should delete existing datasets before creating new ones
        assert "poseest/instance_seg_id" in mock_context.deleted_datasets
        assert "poseest/longterm_seg_id" in mock_context.deleted_datasets

        # Should create new datasets
        assert "poseest/instance_seg_id" in mock_context.created_datasets
        assert "poseest/longterm_seg_id" in mock_context.created_datasets

    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_write_v6_tracklets_single_animal(self, mock_h5py_file):
        """Test writing tracklets for single animal."""
        # Arrange
        pose_file = "test_pose.h5"
        seg_data_shape = (30, 1, 2, 15, 2)  # Single animal
        segmentation_tracks = np.random.randint(1, 5, size=(30, 1), dtype=np.uint32)
        segmentation_ids = np.random.randint(1, 3, size=(30, 1), dtype=np.uint32)

        existing_datasets = ["poseest/seg_data"]
        mock_context = create_mock_h5_context(
            existing_datasets, seg_data_shape=seg_data_shape
        )
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Act
        write_v6_tracklets(pose_file, segmentation_tracks, segmentation_ids)

        # Assert
        # Should successfully create datasets with correct data
        instance_seg_info = mock_context.created_datasets["poseest/instance_seg_id"]
        longterm_seg_info = mock_context.created_datasets["poseest/longterm_seg_id"]

        np.testing.assert_array_equal(
            instance_seg_info["data"], segmentation_tracks.astype(np.uint32)
        )
        np.testing.assert_array_equal(
            longterm_seg_info["data"], segmentation_ids.astype(np.uint32)
        )

    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_write_v6_tracklets_multiple_animals(self, mock_h5py_file):
        """Test writing tracklets for multiple animals."""
        # Arrange
        pose_file = "test_pose.h5"
        seg_data_shape = (200, 5, 3, 20, 2)  # 5 animals
        segmentation_tracks = np.random.randint(0, 15, size=(200, 5), dtype=np.uint32)
        segmentation_ids = np.random.randint(0, 8, size=(200, 5), dtype=np.uint32)

        existing_datasets = ["poseest/seg_data"]
        mock_context = create_mock_h5_context(
            existing_datasets, seg_data_shape=seg_data_shape
        )
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Act
        write_v6_tracklets(pose_file, segmentation_tracks, segmentation_ids)

        # Assert
        # Should successfully handle multiple animals
        assert "poseest/instance_seg_id" in mock_context.created_datasets
        assert "poseest/longterm_seg_id" in mock_context.created_datasets

        instance_seg_info = mock_context.created_datasets["poseest/instance_seg_id"]
        longterm_seg_info = mock_context.created_datasets["poseest/longterm_seg_id"]

        assert instance_seg_info["data"].shape == (200, 5)
        assert longterm_seg_info["data"].shape == (200, 5)


class TestWriteV6TrackletsErrorHandling:
    """Test error handling for write_v6_tracklets."""

    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_missing_segmentation_data_raises_exception(self, mock_h5py_file):
        """Test that missing segmentation data raises InvalidPoseFileException."""
        # Arrange
        pose_file = "test_pose.h5"
        segmentation_tracks = np.zeros((10, 2), dtype=np.uint32)
        segmentation_ids = np.zeros((10, 2), dtype=np.uint32)

        # Mock context without segmentation data
        existing_datasets = []  # No seg_data
        mock_context = create_mock_h5_context(existing_datasets)
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Act & Assert
        with pytest.raises(
            InvalidPoseFileException,
            match="Segmentation data not present in the file",
        ):
            write_v6_tracklets(pose_file, segmentation_tracks, segmentation_ids)

    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_segmentation_tracks_shape_mismatch_raises_exception(self, mock_h5py_file):
        """Test that mismatched segmentation tracks shape raises InvalidPoseFileException."""
        # Arrange
        pose_file = "test_pose.h5"
        seg_data_shape = (100, 3, 2, 10, 2)  # [100 frames, 3 animals]
        segmentation_tracks = np.zeros((100, 2), dtype=np.uint32)  # Wrong animal count
        segmentation_ids = np.zeros((100, 3), dtype=np.uint32)

        existing_datasets = ["poseest/seg_data"]
        mock_context = create_mock_h5_context(
            existing_datasets, seg_data_shape=seg_data_shape
        )
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Act & Assert
        with pytest.raises(
            InvalidPoseFileException,
            match="Segmentation track data does not match segmentation data shape",
        ):
            write_v6_tracklets(pose_file, segmentation_tracks, segmentation_ids)

    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_segmentation_ids_shape_mismatch_raises_exception(self, mock_h5py_file):
        """Test that mismatched segmentation IDs shape raises InvalidPoseFileException."""
        # Arrange
        pose_file = "test_pose.h5"
        seg_data_shape = (75, 4, 1, 5, 2)  # [75 frames, 4 animals]
        segmentation_tracks = np.zeros((75, 4), dtype=np.uint32)
        segmentation_ids = np.zeros((60, 4), dtype=np.uint32)  # Wrong frame count

        existing_datasets = ["poseest/seg_data"]
        mock_context = create_mock_h5_context(
            existing_datasets, seg_data_shape=seg_data_shape
        )
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Act & Assert
        with pytest.raises(
            InvalidPoseFileException,
            match="Segmentation identity data does not match segmentation data shape",
        ):
            write_v6_tracklets(pose_file, segmentation_tracks, segmentation_ids)

    @pytest.mark.parametrize(
        "seg_shape,track_shape,id_shape,expected_error",
        [
            (
                (100, 3),  # seg_data[:2]
                (100, 2),  # wrong animals
                (100, 3),
                "Segmentation track data does not match",
            ),
            (
                (100, 3),  # seg_data[:2]
                (100, 3),
                (90, 3),  # wrong frames
                "Segmentation identity data does not match",
            ),
            (
                (100, 3),  # seg_data[:2]
                (80, 3),  # wrong frames
                (100, 3),
                "Segmentation track data does not match",
            ),
            (
                (100, 3),  # seg_data[:2]
                (100, 4),  # wrong animals
                (100, 4),  # wrong animals (both)
                "Segmentation track data does not match",
            ),
        ],
        ids=[
            "track_animals_mismatch",
            "id_frames_mismatch",
            "track_frames_mismatch",
            "both_animals_mismatch",
        ],
    )
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_various_shape_mismatches(
        self,
        mock_h5py_file,
        seg_shape,
        track_shape,
        id_shape,
        expected_error,
    ):
        """Test various combinations of shape mismatches."""
        # Arrange
        pose_file = "test_pose.h5"
        seg_data_shape = (*seg_shape, 2, 10, 2)  # Add remaining dimensions
        segmentation_tracks = np.zeros(track_shape, dtype=np.uint32)
        segmentation_ids = np.zeros(id_shape, dtype=np.uint32)

        existing_datasets = ["poseest/seg_data"]
        mock_context = create_mock_h5_context(
            existing_datasets, seg_data_shape=seg_data_shape
        )
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Act & Assert
        with pytest.raises(InvalidPoseFileException, match=expected_error):
            write_v6_tracklets(pose_file, segmentation_tracks, segmentation_ids)


class TestWriteV6TrackletsDataTypes:
    """Test data type handling for write_v6_tracklets."""

    @pytest.mark.parametrize(
        "input_dtype,expected_output_dtype",
        [
            (np.int32, np.uint32),
            (np.int64, np.uint32),
            (np.uint16, np.uint32),
            (np.float32, np.uint32),
            (np.float64, np.uint32),
        ],
        ids=["int32", "int64", "uint16", "float32", "float64"],
    )
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_data_type_conversion_tracks(
        self,
        mock_h5py_file,
        input_dtype,
        expected_output_dtype,
    ):
        """Test that segmentation tracks are converted to uint32."""
        # Arrange
        pose_file = "test_pose.h5"
        seg_data_shape = (50, 2, 1, 8, 2)
        segmentation_tracks = np.random.randint(0, 5, size=(50, 2)).astype(input_dtype)
        segmentation_ids = np.random.randint(0, 3, size=(50, 2), dtype=np.uint32)

        existing_datasets = ["poseest/seg_data"]
        mock_context = create_mock_h5_context(
            existing_datasets, seg_data_shape=seg_data_shape
        )
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Act
        write_v6_tracklets(pose_file, segmentation_tracks, segmentation_ids)

        # Assert
        instance_seg_info = mock_context.created_datasets["poseest/instance_seg_id"]
        assert instance_seg_info["data"].dtype == expected_output_dtype

    @pytest.mark.parametrize(
        "input_dtype,expected_output_dtype",
        [
            (np.int32, np.uint32),
            (np.int64, np.uint32),
            (np.uint16, np.uint32),
            (np.float32, np.uint32),
            (np.float64, np.uint32),
        ],
        ids=["int32", "int64", "uint16", "float32", "float64"],
    )
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_data_type_conversion_ids(
        self,
        mock_h5py_file,
        input_dtype,
        expected_output_dtype,
    ):
        """Test that segmentation IDs are converted to uint32."""
        # Arrange
        pose_file = "test_pose.h5"
        seg_data_shape = (40, 3, 1, 6, 2)
        segmentation_tracks = np.random.randint(0, 4, size=(40, 3), dtype=np.uint32)
        segmentation_ids = np.random.randint(0, 2, size=(40, 3)).astype(input_dtype)

        existing_datasets = ["poseest/seg_data"]
        mock_context = create_mock_h5_context(
            existing_datasets, seg_data_shape=seg_data_shape
        )
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Act
        write_v6_tracklets(pose_file, segmentation_tracks, segmentation_ids)

        # Assert
        longterm_seg_info = mock_context.created_datasets["poseest/longterm_seg_id"]
        assert longterm_seg_info["data"].dtype == expected_output_dtype

    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_negative_values_handled_correctly(self, mock_h5py_file):
        """Test handling of negative values in input data."""
        # Arrange
        pose_file = "test_pose.h5"
        seg_data_shape = (20, 2, 1, 5, 2)
        # Include negative values which should be preserved as large uint32 values
        segmentation_tracks = np.array([[-1, 0], [1, -2], [3, 4]], dtype=np.int32)
        segmentation_ids = np.array([[0, -1], [-5, 2], [1, 0]], dtype=np.int32)

        existing_datasets = ["poseest/seg_data"]
        # Adjust seg_data_shape to match the actual data
        seg_data_shape = (3, 2, 1, 5, 2)
        mock_context = create_mock_h5_context(
            existing_datasets, seg_data_shape=seg_data_shape
        )
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Act
        write_v6_tracklets(pose_file, segmentation_tracks, segmentation_ids)

        # Assert
        instance_seg_info = mock_context.created_datasets["poseest/instance_seg_id"]
        longterm_seg_info = mock_context.created_datasets["poseest/longterm_seg_id"]

        # Verify that negative values are converted to their uint32 equivalents
        expected_tracks = segmentation_tracks.astype(np.uint32)
        expected_ids = segmentation_ids.astype(np.uint32)

        np.testing.assert_array_equal(instance_seg_info["data"], expected_tracks)
        np.testing.assert_array_equal(longterm_seg_info["data"], expected_ids)


class TestWriteV6TrackletsEdgeCases:
    """Test edge cases for write_v6_tracklets."""

    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_empty_data_arrays(self, mock_h5py_file):
        """Test handling of empty data arrays."""
        # Arrange
        pose_file = "test_pose.h5"
        seg_data_shape = (0, 0, 1, 5, 2)  # Empty frame and animal dimensions
        segmentation_tracks = np.array([], dtype=np.uint32).reshape(0, 0)
        segmentation_ids = np.array([], dtype=np.uint32).reshape(0, 0)

        existing_datasets = ["poseest/seg_data"]
        mock_context = create_mock_h5_context(
            existing_datasets, seg_data_shape=seg_data_shape
        )
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Act
        write_v6_tracklets(pose_file, segmentation_tracks, segmentation_ids)

        # Assert
        # Should successfully create datasets even with empty data
        assert "poseest/instance_seg_id" in mock_context.created_datasets
        assert "poseest/longterm_seg_id" in mock_context.created_datasets

        instance_seg_info = mock_context.created_datasets["poseest/instance_seg_id"]
        longterm_seg_info = mock_context.created_datasets["poseest/longterm_seg_id"]

        assert instance_seg_info["data"].shape == (0, 0)
        assert longterm_seg_info["data"].shape == (0, 0)

    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_single_frame_data(self, mock_h5py_file):
        """Test handling of single frame data."""
        # Arrange
        pose_file = "test_pose.h5"
        seg_data_shape = (1, 3, 2, 8, 2)  # Single frame
        segmentation_tracks = np.array([[1, 2, 3]], dtype=np.uint32)
        segmentation_ids = np.array([[10, 20, 30]], dtype=np.uint32)

        existing_datasets = ["poseest/seg_data"]
        mock_context = create_mock_h5_context(
            existing_datasets, seg_data_shape=seg_data_shape
        )
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Act
        write_v6_tracklets(pose_file, segmentation_tracks, segmentation_ids)

        # Assert
        instance_seg_info = mock_context.created_datasets["poseest/instance_seg_id"]
        longterm_seg_info = mock_context.created_datasets["poseest/longterm_seg_id"]

        np.testing.assert_array_equal(instance_seg_info["data"], segmentation_tracks)
        np.testing.assert_array_equal(longterm_seg_info["data"], segmentation_ids)

    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_zero_values_data(self, mock_h5py_file):
        """Test handling of all-zero tracklet data."""
        # Arrange
        pose_file = "test_pose.h5"
        seg_data_shape = (50, 2, 1, 10, 2)
        segmentation_tracks = np.zeros((50, 2), dtype=np.uint32)
        segmentation_ids = np.zeros((50, 2), dtype=np.uint32)

        existing_datasets = ["poseest/seg_data"]
        mock_context = create_mock_h5_context(
            existing_datasets, seg_data_shape=seg_data_shape
        )
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Act
        write_v6_tracklets(pose_file, segmentation_tracks, segmentation_ids)

        # Assert
        instance_seg_info = mock_context.created_datasets["poseest/instance_seg_id"]
        longterm_seg_info = mock_context.created_datasets["poseest/longterm_seg_id"]

        np.testing.assert_array_equal(instance_seg_info["data"], segmentation_tracks)
        np.testing.assert_array_equal(longterm_seg_info["data"], segmentation_ids)

    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_max_uint32_values(self, mock_h5py_file):
        """Test handling of maximum uint32 values."""
        # Arrange
        pose_file = "test_pose.h5"
        seg_data_shape = (10, 1, 1, 5, 2)
        max_val = np.iinfo(np.uint32).max
        segmentation_tracks = np.full((10, 1), max_val, dtype=np.uint32)
        segmentation_ids = np.full((10, 1), max_val, dtype=np.uint32)

        existing_datasets = ["poseest/seg_data"]
        mock_context = create_mock_h5_context(
            existing_datasets, seg_data_shape=seg_data_shape
        )
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Act
        write_v6_tracklets(pose_file, segmentation_tracks, segmentation_ids)

        # Assert
        instance_seg_info = mock_context.created_datasets["poseest/instance_seg_id"]
        longterm_seg_info = mock_context.created_datasets["poseest/longterm_seg_id"]

        np.testing.assert_array_equal(instance_seg_info["data"], segmentation_tracks)
        np.testing.assert_array_equal(longterm_seg_info["data"], segmentation_ids)


class TestWriteV6TrackletsIntegration:
    """Integration-style tests for write_v6_tracklets."""

    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_complete_workflow_with_realistic_data(self, mock_h5py_file):
        """Test complete workflow with realistic tracklet data."""
        # Arrange
        pose_file = "realistic_pose.h5"
        num_frames = 1000
        num_animals = 3
        seg_data_shape = (num_frames, num_animals, 2, 15, 2)

        # Create realistic tracklet data with some track changes
        segmentation_tracks = np.zeros((num_frames, num_animals), dtype=np.uint32)
        segmentation_ids = np.zeros((num_frames, num_animals), dtype=np.uint32)

        # Simulate track assignments changing over time
        for frame in range(num_frames):
            for animal in range(num_animals):
                # Simple pattern: tracks cycle every 100 frames
                track_id = (frame // 100) % 5 + 1
                # IDs remain more stable
                identity_id = animal + 1

                segmentation_tracks[frame, animal] = track_id
                segmentation_ids[frame, animal] = identity_id

        existing_datasets = ["poseest/seg_data"]
        mock_context = create_mock_h5_context(
            existing_datasets, seg_data_shape=seg_data_shape
        )
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Act
        write_v6_tracklets(pose_file, segmentation_tracks, segmentation_ids)

        # Assert
        # Verify datasets were created correctly
        assert "poseest/instance_seg_id" in mock_context.created_datasets
        assert "poseest/longterm_seg_id" in mock_context.created_datasets

        instance_seg_info = mock_context.created_datasets["poseest/instance_seg_id"]
        longterm_seg_info = mock_context.created_datasets["poseest/longterm_seg_id"]

        # Verify data integrity
        np.testing.assert_array_equal(
            instance_seg_info["data"], segmentation_tracks.astype(np.uint32)
        )
        np.testing.assert_array_equal(
            longterm_seg_info["data"], segmentation_ids.astype(np.uint32)
        )

        # Verify data properties
        assert instance_seg_info["data"].dtype == np.uint32
        assert longterm_seg_info["data"].dtype == np.uint32
        assert instance_seg_info["data"].shape == (num_frames, num_animals)
        assert longterm_seg_info["data"].shape == (num_frames, num_animals)

    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_workflow_with_dataset_replacement(self, mock_h5py_file):
        """Test workflow where existing datasets are replaced."""
        # Arrange
        pose_file = "test_pose.h5"
        seg_data_shape = (100, 2, 1, 8, 2)
        segmentation_tracks = np.random.randint(1, 10, size=(100, 2), dtype=np.uint32)
        segmentation_ids = np.random.randint(1, 5, size=(100, 2), dtype=np.uint32)

        # Mock existing datasets that will be replaced
        existing_datasets = [
            "poseest/seg_data",
            "poseest/instance_seg_id",
            "poseest/longterm_seg_id",
        ]
        mock_context = create_mock_h5_context(
            existing_datasets, seg_data_shape=seg_data_shape
        )
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Act
        write_v6_tracklets(pose_file, segmentation_tracks, segmentation_ids)

        # Assert
        # Should delete existing datasets
        assert "poseest/instance_seg_id" in mock_context.deleted_datasets
        assert "poseest/longterm_seg_id" in mock_context.deleted_datasets

        # Should create new datasets with correct data
        assert "poseest/instance_seg_id" in mock_context.created_datasets
        assert "poseest/longterm_seg_id" in mock_context.created_datasets

        instance_seg_info = mock_context.created_datasets["poseest/instance_seg_id"]
        longterm_seg_info = mock_context.created_datasets["poseest/longterm_seg_id"]

        np.testing.assert_array_equal(instance_seg_info["data"], segmentation_tracks)
        np.testing.assert_array_equal(longterm_seg_info["data"], segmentation_ids)
