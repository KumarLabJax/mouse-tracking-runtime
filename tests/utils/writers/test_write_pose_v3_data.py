"""Comprehensive unit tests for the write_pose_v3_data function."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mouse_tracking.core.exceptions import InvalidPoseFileException
from mouse_tracking.utils.writers import write_pose_v3_data

from .mock_hdf5 import MockAttrs, create_mock_h5_context


def _create_mock_h5_context(existing_datasets=None):
    """Helper function to create a mock H5 file context manager.

    Args:
        existing_datasets: List of dataset names that already exist in the file

    Returns:
        Mock object that can be used as H5 file context manager
    """
    mock_context = MagicMock()

    # Track created datasets
    created_datasets = {}

    def mock_create_dataset(path, data, **kwargs):
        mock_dataset = MagicMock()
        mock_dataset.attrs = MockAttrs()
        created_datasets[path] = {
            "dataset": mock_dataset,
            "data": data,
            "kwargs": kwargs,
        }
        return mock_dataset

    def mock_getitem(self, key):
        if key in created_datasets:
            return created_datasets[key]["dataset"]
        raise KeyError(f"Dataset {key} not found")

    def mock_contains(self, key):
        return key in (existing_datasets or [])

    def mock_delitem(self, key):
        # Simulate deletion by removing from existing datasets
        pass

    mock_context.create_dataset = mock_create_dataset
    mock_context.__getitem__ = mock_getitem
    mock_context.__contains__ = mock_contains
    mock_context.__delitem__ = mock_delitem
    mock_context.created_datasets = created_datasets

    return mock_context


class TestWritePoseV3DataBasicFunctionality:
    """Test basic functionality of write_pose_v3_data."""

    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_write_all_v3_data_success(self, mock_h5py_file, mock_adjust_pose_version):
        """Test successful writing of all v3 data fields."""
        # Arrange
        pose_file = "test_pose.h5"
        instance_count = np.array([1, 2, 1, 0, 2], dtype=np.uint8)
        instance_embedding = np.random.rand(5, 3, 12).astype(np.float32)
        instance_track = np.array([[0], [1], [0], [0], [2]], dtype=np.uint32)

        mock_context = create_mock_h5_context()
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Act
        write_pose_v3_data(
            pose_file, instance_count, instance_embedding, instance_track
        )

        # Assert
        # Should open file in append mode
        mock_h5py_file.assert_called_once_with(pose_file, "a")

        # Should create all three datasets
        assert "poseest/instance_count" in mock_context.created_datasets
        assert "poseest/instance_embedding" in mock_context.created_datasets
        assert "poseest/instance_track_id" in mock_context.created_datasets

        # Should have correct data types
        count_info = mock_context.created_datasets["poseest/instance_count"]
        np.testing.assert_array_equal(
            count_info["data"], instance_count.astype(np.uint8)
        )

        embed_info = mock_context.created_datasets["poseest/instance_embedding"]
        np.testing.assert_array_equal(
            embed_info["data"], instance_embedding.astype(np.float32)
        )

        track_info = mock_context.created_datasets["poseest/instance_track_id"]
        np.testing.assert_array_equal(
            track_info["data"], instance_track.astype(np.uint32)
        )

        # Should call adjust_pose_version with version 3
        mock_adjust_pose_version.assert_called_once_with(pose_file, 3)

    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_write_partial_v3_data_with_existing_datasets(
        self, mock_h5py_file, mock_adjust_pose_version
    ):
        """Test writing only some v3 data when other datasets already exist."""
        # Arrange
        pose_file = "test_pose.h5"
        instance_count = np.array([2, 1, 0], dtype=np.uint8)
        # Only providing instance_count, others should exist in file

        existing_datasets = ["poseest/instance_embedding", "poseest/instance_track_id"]
        mock_context = create_mock_h5_context(existing_datasets)
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Act
        write_pose_v3_data(pose_file, instance_count, None, None)

        # Assert
        # Should create the provided dataset
        assert "poseest/instance_count" in mock_context.created_datasets
        count_info = mock_context.created_datasets["poseest/instance_count"]
        np.testing.assert_array_equal(
            count_info["data"], instance_count.astype(np.uint8)
        )

        # Should not create the others since they exist
        assert "poseest/instance_embedding" not in mock_context.created_datasets
        assert "poseest/instance_track_id" not in mock_context.created_datasets

        mock_adjust_pose_version.assert_called_once_with(pose_file, 3)

    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_overwrite_existing_v3_datasets(
        self, mock_h5py_file, mock_adjust_pose_version
    ):
        """Test that existing v3 datasets are properly overwritten."""
        # Arrange
        pose_file = "test_pose.h5"
        instance_count = np.array([1, 1, 1], dtype=np.uint8)
        instance_embedding = np.random.rand(3, 2, 12).astype(np.float32)
        instance_track = np.array([[1], [2]], dtype=np.uint32)

        # Mock context with existing datasets
        existing_datasets = [
            "poseest/instance_count",
            "poseest/instance_embedding",
            "poseest/instance_track_id",
        ]
        mock_context = create_mock_h5_context(existing_datasets)
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Track deletions
        deleted_datasets = []

        def track_delitem(self, key):
            deleted_datasets.append(key)

        mock_context.__delitem__ = track_delitem

        # Act
        write_pose_v3_data(
            pose_file, instance_count, instance_embedding, instance_track
        )

        # Assert
        # Should delete existing datasets
        assert "poseest/instance_count" in deleted_datasets
        assert "poseest/instance_embedding" in deleted_datasets
        assert "poseest/instance_track_id" in deleted_datasets

        # Should create new datasets
        assert "poseest/instance_count" in mock_context.created_datasets
        assert "poseest/instance_embedding" in mock_context.created_datasets
        assert "poseest/instance_track_id" in mock_context.created_datasets


class TestWritePoseV3DataErrorHandling:
    """Test error handling in write_pose_v3_data."""

    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_missing_instance_count_not_in_file_raises_exception(self, mock_h5py_file):
        """Test that missing instance_count raises InvalidPoseFileException when not in file."""
        # Arrange
        pose_file = "test_pose.h5"
        instance_embedding = np.random.rand(5, 2, 12).astype(np.float32)
        instance_track = np.array([[1], [2]], dtype=np.uint32)

        mock_context = create_mock_h5_context()  # No existing datasets
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Act & Assert
        with pytest.raises(
            InvalidPoseFileException,
            match="Instance count field was not provided and is required",
        ):
            write_pose_v3_data(pose_file, None, instance_embedding, instance_track)

    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_missing_instance_embedding_not_in_file_raises_exception(
        self, mock_h5py_file
    ):
        """Test that missing instance_embedding raises InvalidPoseFileException when not in file."""
        # Arrange
        pose_file = "test_pose.h5"
        instance_count = np.array([1, 2], dtype=np.uint8)
        instance_track = np.array([[1], [2]], dtype=np.uint32)

        mock_context = create_mock_h5_context()  # No existing datasets
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Act & Assert
        with pytest.raises(
            InvalidPoseFileException,
            match="Instance embedding field was not provided and is required",
        ):
            write_pose_v3_data(pose_file, instance_count, None, instance_track)

    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_missing_instance_track_not_in_file_raises_exception(self, mock_h5py_file):
        """Test that missing instance_track raises InvalidPoseFileException when not in file."""
        # Arrange
        pose_file = "test_pose.h5"
        instance_count = np.array([1, 2], dtype=np.uint8)
        instance_embedding = np.random.rand(5, 2, 12).astype(np.float32)

        mock_context = create_mock_h5_context()  # No existing datasets
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Act & Assert
        with pytest.raises(
            InvalidPoseFileException,
            match="Instance track id field was not provided and is required",
        ):
            write_pose_v3_data(pose_file, instance_count, instance_embedding, None)

    @pytest.mark.parametrize(
        "provided_args,missing_field",
        [
            ((None, "embedding", "track"), "Instance count"),
            (("count", None, "track"), "Instance embedding"),
            (("count", "embedding", None), "Instance track id"),
            ((None, None, "track"), "Instance count"),
            ((None, "embedding", None), "Instance count"),
            (("count", None, None), "Instance embedding"),
            ((None, None, None), "Instance count"),
        ],
        ids=[
            "missing_count",
            "missing_embedding",
            "missing_track",
            "missing_count_and_embedding",
            "missing_count_and_track",
            "missing_embedding_and_track",
            "missing_all",
        ],
    )
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_missing_required_fields_raises_exception(
        self, mock_h5py_file, provided_args, missing_field
    ):
        """Test various combinations of missing required fields."""
        # Arrange
        pose_file = "test_pose.h5"

        # Create dummy data for non-None arguments
        instance_count = np.array([1, 2], dtype=np.uint8) if provided_args[0] else None
        instance_embedding = (
            np.random.rand(2, 1, 12).astype(np.float32) if provided_args[1] else None
        )
        instance_track = np.array([[1]], dtype=np.uint32) if provided_args[2] else None

        mock_context = create_mock_h5_context()  # No existing datasets
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Act & Assert
        with pytest.raises(
            InvalidPoseFileException, match=f"{missing_field}.*was not provided"
        ):
            write_pose_v3_data(
                pose_file, instance_count, instance_embedding, instance_track
            )


class TestWritePoseV3DataDataTypes:
    """Test data type handling in write_pose_v3_data."""

    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_data_type_conversions(self, mock_h5py_file, mock_adjust_pose_version):
        """Test that data is properly converted to required types."""
        # Arrange
        pose_file = "test_pose.h5"
        # Use different input data types
        instance_count = np.array([1, 2, 0], dtype=np.int32)
        instance_embedding = np.random.rand(3, 2, 12).astype(np.float64)
        instance_track = np.array([[1], [2]], dtype=np.int16)

        mock_context = create_mock_h5_context()
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Act
        write_pose_v3_data(
            pose_file, instance_count, instance_embedding, instance_track
        )

        # Assert
        # Should convert instance_count to uint8
        count_info = mock_context.created_datasets["poseest/instance_count"]
        assert count_info["data"].dtype == np.uint8

        # Should convert instance_embedding to float32
        embed_info = mock_context.created_datasets["poseest/instance_embedding"]
        assert embed_info["data"].dtype == np.float32

        # Should convert instance_track to uint32
        track_info = mock_context.created_datasets["poseest/instance_track_id"]
        assert track_info["data"].dtype == np.uint32

    @pytest.mark.parametrize(
        "input_dtype,expected_output_dtype",
        [
            (np.int8, np.uint8),
            (np.int16, np.uint8),
            (np.int32, np.uint8),
            (np.uint16, np.uint8),
            (np.float32, np.uint8),
        ],
        ids=["int8", "int16", "int32", "uint16", "float32"],
    )
    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_instance_count_data_type_conversions(
        self,
        mock_h5py_file,
        mock_adjust_pose_version,
        input_dtype,
        expected_output_dtype,
    ):
        """Test instance_count data type conversions from various input types."""
        # Arrange
        pose_file = "test_pose.h5"
        instance_count = np.array([1, 2, 0], dtype=input_dtype)
        instance_embedding = np.random.rand(3, 2, 12).astype(np.float32)
        instance_track = np.array([[1], [2]], dtype=np.uint32)

        mock_context = create_mock_h5_context()
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Act
        write_pose_v3_data(
            pose_file, instance_count, instance_embedding, instance_track
        )

        # Assert
        count_info = mock_context.created_datasets["poseest/instance_count"]
        assert count_info["data"].dtype == expected_output_dtype

    @pytest.mark.parametrize(
        "input_dtype,expected_output_dtype",
        [
            (np.float16, np.float32),
            (np.float64, np.float32),
            (np.int32, np.float32),
        ],
        ids=["float16", "float64", "int32"],
    )
    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_instance_embedding_data_type_conversions(
        self,
        mock_h5py_file,
        mock_adjust_pose_version,
        input_dtype,
        expected_output_dtype,
    ):
        """Test instance_embedding data type conversions from various input types."""
        # Arrange
        pose_file = "test_pose.h5"
        instance_count = np.array([1, 2], dtype=np.uint8)
        instance_embedding = np.random.rand(2, 2, 12).astype(input_dtype)
        instance_track = np.array([[1], [2]], dtype=np.uint32)

        mock_context = create_mock_h5_context()
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Act
        write_pose_v3_data(
            pose_file, instance_count, instance_embedding, instance_track
        )

        # Assert
        embed_info = mock_context.created_datasets["poseest/instance_embedding"]
        assert embed_info["data"].dtype == expected_output_dtype

    @pytest.mark.parametrize(
        "input_dtype,expected_output_dtype",
        [
            (np.int8, np.uint32),
            (np.int16, np.uint32),
            (np.int32, np.uint32),
            (np.uint8, np.uint32),
            (np.uint16, np.uint32),
        ],
        ids=["int8", "int16", "int32", "uint8", "uint16"],
    )
    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_instance_track_data_type_conversions(
        self,
        mock_h5py_file,
        mock_adjust_pose_version,
        input_dtype,
        expected_output_dtype,
    ):
        """Test instance_track data type conversions from various input types."""
        # Arrange
        pose_file = "test_pose.h5"
        instance_count = np.array([1, 2], dtype=np.uint8)
        instance_embedding = np.random.rand(2, 2, 12).astype(np.float32)
        instance_track = np.array([[1], [2]], dtype=input_dtype)

        mock_context = create_mock_h5_context()
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Act
        write_pose_v3_data(
            pose_file, instance_count, instance_embedding, instance_track
        )

        # Assert
        track_info = mock_context.created_datasets["poseest/instance_track_id"]
        assert track_info["data"].dtype == expected_output_dtype


class TestWritePoseV3DataVersionHandling:
    """Test version handling logic in write_pose_v3_data."""

    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_always_calls_version_3(self, mock_h5py_file, mock_adjust_pose_version):
        """Test that the function always calls adjust_pose_version with version 3."""
        # Arrange
        pose_file = "test_pose.h5"
        instance_count = np.array([1], dtype=np.uint8)
        instance_embedding = np.random.rand(1, 1, 12).astype(np.float32)
        instance_track = np.array([[1]], dtype=np.uint32)

        mock_context = create_mock_h5_context()
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Act
        write_pose_v3_data(
            pose_file, instance_count, instance_embedding, instance_track
        )

        # Assert
        mock_adjust_pose_version.assert_called_once_with(pose_file, 3)

    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_version_called_even_with_existing_data(
        self, mock_h5py_file, mock_adjust_pose_version
    ):
        """Test that version is called even when no new datasets are created."""
        # Arrange
        pose_file = "test_pose.h5"

        # All datasets already exist
        existing_datasets = [
            "poseest/instance_count",
            "poseest/instance_embedding",
            "poseest/instance_track_id",
        ]
        mock_context = create_mock_h5_context(existing_datasets)
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Act
        write_pose_v3_data(pose_file, None, None, None)

        # Assert
        mock_adjust_pose_version.assert_called_once_with(pose_file, 3)


class TestWritePoseV3DataEdgeCases:
    """Test edge cases and boundary conditions of write_pose_v3_data."""

    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_empty_data_arrays(self, mock_h5py_file, mock_adjust_pose_version):
        """Test handling of empty data arrays."""
        # Arrange
        pose_file = "test_pose.h5"
        instance_count = np.empty((0,), dtype=np.uint8)
        instance_embedding = np.empty((0, 0, 12), dtype=np.float32)
        instance_track = np.empty((0, 0), dtype=np.uint32)

        mock_context = create_mock_h5_context()
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Act
        write_pose_v3_data(
            pose_file, instance_count, instance_embedding, instance_track
        )

        # Assert
        # Should still create datasets even with empty data
        assert "poseest/instance_count" in mock_context.created_datasets
        assert "poseest/instance_embedding" in mock_context.created_datasets
        assert "poseest/instance_track_id" in mock_context.created_datasets
        mock_adjust_pose_version.assert_called_once_with(pose_file, 3)

    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_single_frame_data(self, mock_h5py_file, mock_adjust_pose_version):
        """Test handling of single frame data."""
        # Arrange
        pose_file = "test_pose.h5"
        instance_count = np.array([2], dtype=np.uint8)
        instance_embedding = np.random.rand(1, 2, 12).astype(np.float32)
        instance_track = np.array([[1, 2]], dtype=np.uint32)

        mock_context = create_mock_h5_context()
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Act
        write_pose_v3_data(
            pose_file, instance_count, instance_embedding, instance_track
        )

        # Assert
        count_info = mock_context.created_datasets["poseest/instance_count"]
        assert count_info["data"].shape == (1,)

        embed_info = mock_context.created_datasets["poseest/instance_embedding"]
        assert embed_info["data"].shape == (1, 2, 12)

        track_info = mock_context.created_datasets["poseest/instance_track_id"]
        assert track_info["data"].shape == (1, 2)

        mock_adjust_pose_version.assert_called_once_with(pose_file, 3)

    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_large_multi_animal_data(self, mock_h5py_file, mock_adjust_pose_version):
        """Test handling of large multi-animal datasets."""
        # Arrange
        pose_file = "test_pose.h5"
        num_frames = 10000
        num_animals = 10

        instance_count = np.random.randint(
            0, num_animals + 1, size=num_frames, dtype=np.uint8
        )
        instance_embedding = np.random.rand(num_frames, num_animals, 12).astype(
            np.float32
        )
        instance_track = np.random.randint(
            0, 100, size=(num_frames, num_animals), dtype=np.uint32
        )

        mock_context = create_mock_h5_context()
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Act
        write_pose_v3_data(
            pose_file, instance_count, instance_embedding, instance_track
        )

        # Assert
        count_info = mock_context.created_datasets["poseest/instance_count"]
        assert count_info["data"].shape == (num_frames,)

        embed_info = mock_context.created_datasets["poseest/instance_embedding"]
        assert embed_info["data"].shape == (num_frames, num_animals, 12)

        track_info = mock_context.created_datasets["poseest/instance_track_id"]
        assert track_info["data"].shape == (num_frames, num_animals)

        mock_adjust_pose_version.assert_called_once_with(pose_file, 3)


class TestWritePoseV3DataIntegration:
    """Test integration scenarios for write_pose_v3_data."""

    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_complete_workflow_new_datasets(
        self, mock_h5py_file, mock_adjust_pose_version
    ):
        """Test complete workflow for creating new v3 datasets."""
        # Arrange
        pose_file = "/path/to/pose_v3.h5"
        num_frames = 1000
        num_animals = 3

        instance_count = np.random.randint(
            0, num_animals + 1, size=num_frames, dtype=np.uint8
        )
        instance_embedding = np.random.rand(num_frames, num_animals, 12).astype(
            np.float32
        )
        instance_track = np.random.randint(
            0, 50, size=(num_frames, num_animals), dtype=np.uint32
        )

        mock_context = create_mock_h5_context()  # No existing datasets
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Act
        write_pose_v3_data(
            pose_file, instance_count, instance_embedding, instance_track
        )

        # Assert
        # Should open file correctly
        mock_h5py_file.assert_called_once_with(pose_file, "a")

        # Should create all three datasets with correct data
        assert "poseest/instance_count" in mock_context.created_datasets
        assert "poseest/instance_embedding" in mock_context.created_datasets
        assert "poseest/instance_track_id" in mock_context.created_datasets

        # Verify data shapes and types
        count_info = mock_context.created_datasets["poseest/instance_count"]
        assert count_info["data"].shape == (num_frames,)
        assert count_info["data"].dtype == np.uint8

        embed_info = mock_context.created_datasets["poseest/instance_embedding"]
        assert embed_info["data"].shape == (num_frames, num_animals, 12)
        assert embed_info["data"].dtype == np.float32

        track_info = mock_context.created_datasets["poseest/instance_track_id"]
        assert track_info["data"].shape == (num_frames, num_animals)
        assert track_info["data"].dtype == np.uint32

        # Should call version adjustment
        mock_adjust_pose_version.assert_called_once_with(pose_file, 3)

    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_complete_workflow_overwrite_existing(
        self, mock_h5py_file, mock_adjust_pose_version
    ):
        """Test complete workflow for overwriting existing v3 datasets."""
        # Arrange
        pose_file = "/path/to/existing_pose_v3.h5"
        instance_count = np.array([2, 1, 3], dtype=np.uint8)
        instance_embedding = np.random.rand(3, 3, 12).astype(np.float32)
        instance_track = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint32)

        # All datasets already exist
        existing_datasets = [
            "poseest/instance_count",
            "poseest/instance_embedding",
            "poseest/instance_track_id",
        ]
        mock_context = create_mock_h5_context(existing_datasets)
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Track deletions
        deleted_datasets = []

        def track_delitem(self, key):
            deleted_datasets.append(key)

        mock_context.__delitem__ = track_delitem

        # Act
        write_pose_v3_data(
            pose_file, instance_count, instance_embedding, instance_track
        )

        # Assert
        # Should delete all existing datasets
        assert "poseest/instance_count" in deleted_datasets
        assert "poseest/instance_embedding" in deleted_datasets
        assert "poseest/instance_track_id" in deleted_datasets

        # Should create all new datasets
        assert "poseest/instance_count" in mock_context.created_datasets
        assert "poseest/instance_embedding" in mock_context.created_datasets
        assert "poseest/instance_track_id" in mock_context.created_datasets

        # Should call version adjustment
        mock_adjust_pose_version.assert_called_once_with(pose_file, 3)

    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_mixed_workflow_some_existing_some_new(
        self, mock_h5py_file, mock_adjust_pose_version
    ):
        """Test workflow with some existing and some new datasets."""
        # Arrange
        pose_file = "/path/to/mixed_pose_v3.h5"
        instance_count = np.array([1, 2], dtype=np.uint8)
        instance_embedding = np.random.rand(2, 2, 12).astype(np.float32)
        instance_track = np.array([[1, 2], [3, 4]], dtype=np.uint32)

        # Only instance_count exists
        existing_datasets = ["poseest/instance_count"]
        mock_context = create_mock_h5_context(existing_datasets)
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        deleted_datasets = []

        def track_delitem(self, key):
            deleted_datasets.append(key)

        mock_context.__delitem__ = track_delitem

        # Act
        write_pose_v3_data(
            pose_file, instance_count, instance_embedding, instance_track
        )

        # Assert
        # Should delete existing instance_count
        assert "poseest/instance_count" in deleted_datasets

        # Should create all three datasets (including overwritten instance_count)
        assert "poseest/instance_count" in mock_context.created_datasets
        assert "poseest/instance_embedding" in mock_context.created_datasets
        assert "poseest/instance_track_id" in mock_context.created_datasets

        mock_adjust_pose_version.assert_called_once_with(pose_file, 3)
