"""Tests for the write_pose_v4_data function in mouse_tracking.utils.writers."""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from mouse_tracking.core.exceptions import InvalidPoseFileException
from mouse_tracking.utils.writers import write_pose_v4_data

from .mock_hdf5 import create_mock_h5_context


class TestWritePoseV4DataBasicFunctionality:
    """Test basic functionality and success cases for write_pose_v4_data."""

    @patch("mouse_tracking.utils.writers.h5py.File")
    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    def test_write_all_v4_data_success(self, mock_adjust, mock_h5_file):
        """Test successful writing of all v4 data fields."""
        # Arrange
        mock_file = create_mock_h5_context()
        mock_h5_file.return_value = mock_file

        pose_file = "test.h5"
        mask = np.array([[True, False], [False, True]], dtype=bool)
        longterm_ids = np.array([[1, 2], [2, 1]], dtype=np.uint32)
        centers = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
        embeddings = np.random.random((2, 2, 128)).astype(np.float32)

        # Act
        write_pose_v4_data(pose_file, mask, longterm_ids, centers, embeddings)

        # Assert
        mock_h5_file.assert_called_once_with(pose_file, "a")
        mock_adjust.assert_called_once_with(pose_file, 4)

        # Verify dataset creation calls
        assert mock_file.create_dataset.call_count == 4
        created_datasets = [
            call[0][0] for call in mock_file.create_dataset.call_args_list
        ]
        expected_datasets = [
            "poseest/id_mask",
            "poseest/instance_embed_id",
            "poseest/instance_id_center",
            "poseest/identity_embeds",
        ]
        assert set(created_datasets) == set(expected_datasets)

    @patch("mouse_tracking.utils.writers.h5py.File")
    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    def test_write_v4_data_without_embeddings_existing_in_file(
        self, mock_adjust, mock_h5_file
    ):
        """Test writing v4 data without embeddings parameter when embeddings exist in file."""
        # Arrange
        mock_file = create_mock_h5_context()
        mock_file._datasets["poseest/identity_embeds"] = Mock()
        mock_h5_file.return_value = mock_file

        pose_file = "test.h5"
        mask = np.array([[True, False]], dtype=bool)
        longterm_ids = np.array([[1, 2]], dtype=np.uint32)
        centers = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)

        # Act
        write_pose_v4_data(pose_file, mask, longterm_ids, centers)

        # Assert
        mock_h5_file.assert_called_once_with(pose_file, "a")
        mock_adjust.assert_called_once_with(pose_file, 4)

        # Verify only 3 datasets created (no embeddings)
        assert mock_file.create_dataset.call_count == 3
        created_datasets = [
            call[0][0] for call in mock_file.create_dataset.call_args_list
        ]
        expected_datasets = [
            "poseest/id_mask",
            "poseest/instance_embed_id",
            "poseest/instance_id_center",
        ]
        assert set(created_datasets) == set(expected_datasets)
        assert "poseest/identity_embeds" not in created_datasets

    @patch("mouse_tracking.utils.writers.h5py.File")
    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    def test_overwrite_existing_v4_datasets(self, mock_adjust, mock_h5_file):
        """Test that existing v4 datasets are properly deleted and recreated."""
        # Arrange
        mock_file = create_mock_h5_context()
        # Simulate existing datasets
        mock_file._datasets = {
            "poseest/id_mask": Mock(),
            "poseest/instance_embed_id": Mock(),
            "poseest/instance_id_center": Mock(),
            "poseest/identity_embeds": Mock(),
        }
        mock_h5_file.return_value = mock_file

        pose_file = "test.h5"
        mask = np.array([[True]], dtype=bool)
        longterm_ids = np.array([[1]], dtype=np.uint32)
        centers = np.array([[0.1, 0.2]], dtype=np.float64)
        embeddings = np.random.random((1, 1, 128)).astype(np.float32)

        # Act
        write_pose_v4_data(pose_file, mask, longterm_ids, centers, embeddings)

        # Assert
        # Verify all existing datasets were deleted
        assert mock_file.__delitem__.call_count == 4
        deleted_datasets = [call[0][0] for call in mock_file.__delitem__.call_args_list]
        expected_deletions = [
            "poseest/id_mask",
            "poseest/instance_embed_id",
            "poseest/instance_id_center",
            "poseest/identity_embeds",
        ]
        assert set(deleted_datasets) == set(expected_deletions)


class TestWritePoseV4DataErrorHandling:
    """Test error handling scenarios for write_pose_v4_data."""

    @patch("mouse_tracking.utils.writers.h5py.File")
    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    def test_missing_embeddings_not_in_file_raises_exception(
        self, mock_adjust, mock_h5_file
    ):
        """Test that missing embeddings when not in file raises InvalidPoseFileException."""
        # Arrange
        mock_file = create_mock_h5_context()
        mock_h5_file.return_value = mock_file

        pose_file = "test.h5"
        mask = np.array([[True, False]], dtype=bool)
        longterm_ids = np.array([[1, 2]], dtype=np.uint32)
        centers = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)

        # Act & Assert
        with pytest.raises(
            InvalidPoseFileException,
            match="Identity embedding values not provided and is required",
        ):
            write_pose_v4_data(pose_file, mask, longterm_ids, centers)

        # Verify adjust_pose_version was not called due to exception
        mock_adjust.assert_not_called()


class TestWritePoseV4DataDataTypes:
    """Test data type conversions for write_pose_v4_data."""

    @patch("mouse_tracking.utils.writers.h5py.File")
    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    def test_data_type_conversions(self, mock_adjust, mock_h5_file):
        """Test that all data types are converted correctly."""
        # Arrange
        mock_file = create_mock_h5_context()
        mock_h5_file.return_value = mock_file

        pose_file = "test.h5"
        mask = np.array([[1, 0], [0, 1]], dtype=np.int32)  # Will be converted to bool
        longterm_ids = np.array(
            [[1.0, 2.0], [2.0, 1.0]], dtype=np.float64
        )  # Will be converted to uint32
        centers = np.array(
            [[0.1, 0.2], [0.3, 0.4]], dtype=np.float32
        )  # Will be converted to float64
        embeddings = np.random.random((2, 2, 128)).astype(
            np.float64
        )  # Will be converted to float32

        # Act
        write_pose_v4_data(pose_file, mask, longterm_ids, centers, embeddings)

        # Assert
        # Verify create_dataset was called with correct data types
        create_calls = mock_file.create_dataset.call_args_list

        # Check mask conversion to bool
        mask_call = next(
            call for call in create_calls if call[0][0] == "poseest/id_mask"
        )
        assert mask_call[1]["data"].dtype == bool

        # Check longterm_ids conversion to uint32
        ids_call = next(
            call for call in create_calls if call[0][0] == "poseest/instance_embed_id"
        )
        assert ids_call[1]["data"].dtype == np.uint32

        # Check centers conversion to float64
        centers_call = next(
            call for call in create_calls if call[0][0] == "poseest/instance_id_center"
        )
        assert centers_call[1]["data"].dtype == np.float64

        # Check embeddings conversion to float32
        embeds_call = next(
            call for call in create_calls if call[0][0] == "poseest/identity_embeds"
        )
        assert embeds_call[1]["data"].dtype == np.float32

    @pytest.mark.parametrize(
        "input_dtype", [np.uint8, np.int8, np.int16, np.int32, np.float32, np.float64]
    )
    @patch("mouse_tracking.utils.writers.h5py.File")
    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    def test_mask_data_type_conversions(self, mock_adjust, mock_h5_file, input_dtype):
        """Test mask data type conversion from various input types."""
        # Arrange
        mock_file = create_mock_h5_context()
        mock_h5_file.return_value = mock_file

        pose_file = "test.h5"
        mask = np.array([[1, 0], [0, 1]], dtype=input_dtype)
        longterm_ids = np.array([[1, 2], [2, 1]], dtype=np.uint32)
        centers = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
        embeddings = np.random.random((2, 2, 128)).astype(np.float32)

        # Act
        write_pose_v4_data(pose_file, mask, longterm_ids, centers, embeddings)

        # Assert
        create_calls = mock_file.create_dataset.call_args_list
        mask_call = next(
            call for call in create_calls if call[0][0] == "poseest/id_mask"
        )
        assert mask_call[1]["data"].dtype == bool

    @pytest.mark.parametrize(
        "input_dtype",
        [np.int8, np.int16, np.int32, np.uint8, np.uint16, np.float32, np.float64],
    )
    @patch("mouse_tracking.utils.writers.h5py.File")
    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    def test_longterm_ids_data_type_conversions(
        self, mock_adjust, mock_h5_file, input_dtype
    ):
        """Test longterm_ids data type conversion from various input types."""
        # Arrange
        mock_file = create_mock_h5_context()
        mock_h5_file.return_value = mock_file

        pose_file = "test.h5"
        mask = np.array([[True, False], [False, True]], dtype=bool)
        longterm_ids = np.array([[1, 2], [2, 1]], dtype=input_dtype)
        centers = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
        embeddings = np.random.random((2, 2, 128)).astype(np.float32)

        # Act
        write_pose_v4_data(pose_file, mask, longterm_ids, centers, embeddings)

        # Assert
        create_calls = mock_file.create_dataset.call_args_list
        ids_call = next(
            call for call in create_calls if call[0][0] == "poseest/instance_embed_id"
        )
        assert ids_call[1]["data"].dtype == np.uint32

    @pytest.mark.parametrize(
        "input_dtype", [np.float16, np.float32, np.int32, np.int64]
    )
    @patch("mouse_tracking.utils.writers.h5py.File")
    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    def test_centers_data_type_conversions(
        self, mock_adjust, mock_h5_file, input_dtype
    ):
        """Test centers data type conversion from various input types."""
        # Arrange
        mock_file = create_mock_h5_context()
        mock_h5_file.return_value = mock_file

        pose_file = "test.h5"
        mask = np.array([[True, False], [False, True]], dtype=bool)
        longterm_ids = np.array([[1, 2], [2, 1]], dtype=np.uint32)
        centers = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=input_dtype)
        embeddings = np.random.random((2, 2, 128)).astype(np.float32)

        # Act
        write_pose_v4_data(pose_file, mask, longterm_ids, centers, embeddings)

        # Assert
        create_calls = mock_file.create_dataset.call_args_list
        centers_call = next(
            call for call in create_calls if call[0][0] == "poseest/instance_id_center"
        )
        assert centers_call[1]["data"].dtype == np.float64

    @pytest.mark.parametrize("input_dtype", [np.float16, np.float64, np.int32])
    @patch("mouse_tracking.utils.writers.h5py.File")
    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    def test_embeddings_data_type_conversions(
        self, mock_adjust, mock_h5_file, input_dtype
    ):
        """Test embeddings data type conversion from various input types."""
        # Arrange
        mock_file = create_mock_h5_context()
        mock_h5_file.return_value = mock_file

        pose_file = "test.h5"
        mask = np.array([[True, False], [False, True]], dtype=bool)
        longterm_ids = np.array([[1, 2], [2, 1]], dtype=np.uint32)
        centers = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
        embeddings = np.random.random((2, 2, 128)).astype(input_dtype)

        # Act
        write_pose_v4_data(pose_file, mask, longterm_ids, centers, embeddings)

        # Assert
        create_calls = mock_file.create_dataset.call_args_list
        embeds_call = next(
            call for call in create_calls if call[0][0] == "poseest/identity_embeds"
        )
        assert embeds_call[1]["data"].dtype == np.float32


class TestWritePoseV4DataVersionHandling:
    """Test version handling for write_pose_v4_data."""

    @patch("mouse_tracking.utils.writers.h5py.File")
    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    def test_always_calls_version_4(self, mock_adjust, mock_h5_file):
        """Test that adjust_pose_version is always called with version 4."""
        # Arrange
        mock_file = create_mock_h5_context()
        mock_h5_file.return_value = mock_file

        pose_file = "test.h5"
        mask = np.array([[True]], dtype=bool)
        longterm_ids = np.array([[1]], dtype=np.uint32)
        centers = np.array([[0.1, 0.2]], dtype=np.float64)
        embeddings = np.random.random((1, 1, 128)).astype(np.float32)

        # Act
        write_pose_v4_data(pose_file, mask, longterm_ids, centers, embeddings)

        # Assert
        mock_adjust.assert_called_once_with(pose_file, 4)

    @patch("mouse_tracking.utils.writers.h5py.File")
    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    def test_version_called_even_with_existing_data(self, mock_adjust, mock_h5_file):
        """Test that version is adjusted even when some datasets already exist."""
        # Arrange
        mock_file = create_mock_h5_context()
        mock_file._datasets = {
            "poseest/id_mask": Mock(),
            "poseest/instance_embed_id": Mock(),
        }
        mock_h5_file.return_value = mock_file

        pose_file = "test.h5"
        mask = np.array([[True]], dtype=bool)
        longterm_ids = np.array([[1]], dtype=np.uint32)
        centers = np.array([[0.1, 0.2]], dtype=np.float64)
        embeddings = np.random.random((1, 1, 128)).astype(np.float32)

        # Act
        write_pose_v4_data(pose_file, mask, longterm_ids, centers, embeddings)

        # Assert
        mock_adjust.assert_called_once_with(pose_file, 4)


class TestWritePoseV4DataEdgeCases:
    """Test edge cases for write_pose_v4_data."""

    @patch("mouse_tracking.utils.writers.h5py.File")
    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    def test_empty_data_arrays(self, mock_adjust, mock_h5_file):
        """Test handling of empty data arrays."""
        # Arrange
        mock_file = create_mock_h5_context()
        mock_h5_file.return_value = mock_file

        pose_file = "test.h5"
        mask = np.array([], dtype=bool).reshape(0, 2)
        longterm_ids = np.array([], dtype=np.uint32).reshape(0, 2)
        centers = np.array([], dtype=np.float64).reshape(0, 2)
        embeddings = np.array([], dtype=np.float32).reshape(0, 2, 128)

        # Act
        write_pose_v4_data(pose_file, mask, longterm_ids, centers, embeddings)

        # Assert
        mock_h5_file.assert_called_once_with(pose_file, "a")
        mock_adjust.assert_called_once_with(pose_file, 4)
        assert mock_file.create_dataset.call_count == 4

    @patch("mouse_tracking.utils.writers.h5py.File")
    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    def test_single_frame_single_animal(self, mock_adjust, mock_h5_file):
        """Test handling of single frame, single animal data."""
        # Arrange
        mock_file = create_mock_h5_context()
        mock_h5_file.return_value = mock_file

        pose_file = "test.h5"
        mask = np.array([[True]], dtype=bool)
        longterm_ids = np.array([[1]], dtype=np.uint32)
        centers = np.array([[0.1, 0.2]], dtype=np.float64)
        embeddings = np.random.random((1, 1, 128)).astype(np.float32)

        # Act
        write_pose_v4_data(pose_file, mask, longterm_ids, centers, embeddings)

        # Assert
        mock_h5_file.assert_called_once_with(pose_file, "a")
        mock_adjust.assert_called_once_with(pose_file, 4)
        assert mock_file.create_dataset.call_count == 4

    @patch("mouse_tracking.utils.writers.h5py.File")
    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    def test_large_multi_animal_data(self, mock_adjust, mock_h5_file):
        """Test handling of large multi-animal datasets."""
        # Arrange
        mock_file = create_mock_h5_context()
        mock_h5_file.return_value = mock_file

        pose_file = "test.h5"
        n_frames, n_animals, embed_dim = 1000, 5, 256
        mask = np.random.choice([True, False], size=(n_frames, n_animals))
        longterm_ids = np.random.randint(
            0, 10, size=(n_frames, n_animals), dtype=np.uint32
        )
        centers = np.random.random((10, embed_dim)).astype(np.float64)
        embeddings = np.random.random((n_frames, n_animals, embed_dim)).astype(
            np.float32
        )

        # Act
        write_pose_v4_data(pose_file, mask, longterm_ids, centers, embeddings)

        # Assert
        mock_h5_file.assert_called_once_with(pose_file, "a")
        mock_adjust.assert_called_once_with(pose_file, 4)
        assert mock_file.create_dataset.call_count == 4


class TestWritePoseV4DataIntegration:
    """Test integration scenarios for write_pose_v4_data."""

    @patch("mouse_tracking.utils.writers.h5py.File")
    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    def test_complete_workflow_new_datasets(self, mock_adjust, mock_h5_file):
        """Test complete workflow with new datasets (none exist)."""
        # Arrange
        mock_file = create_mock_h5_context()
        mock_h5_file.return_value = mock_file

        pose_file = "test.h5"
        mask = np.array([[True, False], [False, True]], dtype=bool)
        longterm_ids = np.array([[1, 2], [2, 1]], dtype=np.uint32)
        centers = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
        embeddings = np.random.random((2, 2, 128)).astype(np.float32)

        # Act
        write_pose_v4_data(pose_file, mask, longterm_ids, centers, embeddings)

        # Assert
        mock_h5_file.assert_called_once_with(pose_file, "a")
        mock_adjust.assert_called_once_with(pose_file, 4)

        # Verify no deletions occurred (no existing datasets)
        assert mock_file.__delitem__.call_count == 0

        # Verify all 4 datasets created
        assert mock_file.create_dataset.call_count == 4
        created_datasets = [
            call[0][0] for call in mock_file.create_dataset.call_args_list
        ]
        expected_datasets = [
            "poseest/id_mask",
            "poseest/instance_embed_id",
            "poseest/instance_id_center",
            "poseest/identity_embeds",
        ]
        assert set(created_datasets) == set(expected_datasets)

    @patch("mouse_tracking.utils.writers.h5py.File")
    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    def test_complete_workflow_overwrite_existing(self, mock_adjust, mock_h5_file):
        """Test complete workflow when all datasets already exist."""
        # Arrange
        mock_file = create_mock_h5_context()
        mock_file._datasets = {
            "poseest/id_mask": Mock(),
            "poseest/instance_embed_id": Mock(),
            "poseest/instance_id_center": Mock(),
            "poseest/identity_embeds": Mock(),
        }
        mock_h5_file.return_value = mock_file

        pose_file = "test.h5"
        mask = np.array([[True, False]], dtype=bool)
        longterm_ids = np.array([[1, 2]], dtype=np.uint32)
        centers = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
        embeddings = np.random.random((1, 2, 128)).astype(np.float32)

        # Act
        write_pose_v4_data(pose_file, mask, longterm_ids, centers, embeddings)

        # Assert
        mock_h5_file.assert_called_once_with(pose_file, "a")
        mock_adjust.assert_called_once_with(pose_file, 4)

        # Verify all existing datasets were deleted
        assert mock_file.__delitem__.call_count == 4
        deleted_datasets = [call[0][0] for call in mock_file.__delitem__.call_args_list]
        expected_deletions = [
            "poseest/id_mask",
            "poseest/instance_embed_id",
            "poseest/instance_id_center",
            "poseest/identity_embeds",
        ]
        assert set(deleted_datasets) == set(expected_deletions)

        # Verify all datasets recreated
        assert mock_file.create_dataset.call_count == 4

    @patch("mouse_tracking.utils.writers.h5py.File")
    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    def test_mixed_workflow_some_existing_some_new(self, mock_adjust, mock_h5_file):
        """Test workflow when some datasets exist and some are new."""
        # Arrange
        mock_file = create_mock_h5_context()
        mock_file._datasets = {
            "poseest/id_mask": Mock(),
            "poseest/instance_id_center": Mock(),
        }
        mock_h5_file.return_value = mock_file

        pose_file = "test.h5"
        mask = np.array([[True, False]], dtype=bool)
        longterm_ids = np.array([[1, 2]], dtype=np.uint32)
        centers = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
        embeddings = np.random.random((1, 2, 128)).astype(np.float32)

        # Act
        write_pose_v4_data(pose_file, mask, longterm_ids, centers, embeddings)

        # Assert
        mock_h5_file.assert_called_once_with(pose_file, "a")
        mock_adjust.assert_called_once_with(pose_file, 4)

        # Verify only existing datasets were deleted
        assert mock_file.__delitem__.call_count == 2
        deleted_datasets = [call[0][0] for call in mock_file.__delitem__.call_args_list]
        expected_deletions = ["poseest/id_mask", "poseest/instance_id_center"]
        assert set(deleted_datasets) == set(expected_deletions)

        # Verify all 4 datasets created (including recreating deleted ones)
        assert mock_file.create_dataset.call_count == 4

    @patch("mouse_tracking.utils.writers.h5py.File")
    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    def test_workflow_without_embeddings_param_but_existing_in_file(
        self, mock_adjust, mock_h5_file
    ):
        """Test workflow without embeddings parameter when embeddings exist in file."""
        # Arrange
        mock_file = create_mock_h5_context()
        mock_file._datasets = {
            "poseest/identity_embeds": Mock(),
            "poseest/id_mask": Mock(),
        }
        mock_h5_file.return_value = mock_file

        pose_file = "test.h5"
        mask = np.array([[True, False]], dtype=bool)
        longterm_ids = np.array([[1, 2]], dtype=np.uint32)
        centers = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)

        # Act
        write_pose_v4_data(pose_file, mask, longterm_ids, centers)

        # Assert
        mock_h5_file.assert_called_once_with(pose_file, "a")
        mock_adjust.assert_called_once_with(pose_file, 4)

        # Verify only non-embedding datasets were deleted
        assert mock_file.__delitem__.call_count == 1
        deleted_datasets = [call[0][0] for call in mock_file.__delitem__.call_args_list]
        assert "poseest/id_mask" in deleted_datasets
        assert "poseest/identity_embeds" not in deleted_datasets

        # Verify only 3 datasets created (no embeddings)
        assert mock_file.create_dataset.call_count == 3
        created_datasets = [
            call[0][0] for call in mock_file.create_dataset.call_args_list
        ]
        expected_datasets = [
            "poseest/id_mask",
            "poseest/instance_embed_id",
            "poseest/instance_id_center",
        ]
        assert set(created_datasets) == set(expected_datasets)
        assert "poseest/identity_embeds" not in created_datasets
