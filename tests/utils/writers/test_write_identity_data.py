"""Comprehensive unit tests for the write_identity_data function."""

from unittest.mock import patch

import numpy as np
import pytest

from mouse_tracking.core.exceptions import InvalidPoseFileException
from mouse_tracking.utils.writers import write_identity_data

from .mock_hdf5 import create_mock_h5_context


class TestWriteIdentityDataBasicFunctionality:
    """Test basic functionality of write_identity_data."""

    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_write_identity_data_success(
        self, mock_h5py_file, mock_adjust_pose_version
    ):
        """Test successful writing of identity data."""
        # Arrange
        pose_file = "test_pose.h5"
        pose_data_shape = (100, 3, 12, 2)  # [frame, num_animals, keypoints, coords]
        embeddings = np.random.rand(100, 3, 128).astype(
            np.float32
        )  # [frame, num_animals, embed_dim]
        config_str = "test_config"
        model_str = "test_model"

        existing_datasets = ["poseest/points"]
        mock_context = create_mock_h5_context(existing_datasets, pose_data_shape)
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Act
        write_identity_data(pose_file, embeddings, config_str, model_str)

        # Assert
        # Should call adjust_pose_version first
        mock_adjust_pose_version.assert_called_once_with(pose_file, 4)

        # Should open file in append mode
        mock_h5py_file.assert_called_once_with(pose_file, "a")

        # Should create identity_embeds dataset
        assert "poseest/identity_embeds" in mock_context.created_datasets
        identity_info = mock_context.created_datasets["poseest/identity_embeds"]
        np.testing.assert_array_equal(
            identity_info["data"], embeddings.astype(np.float32)
        )

        # Should set attributes on the dataset
        identity_dataset = identity_info["dataset"]
        assert identity_dataset.attrs["config"] == config_str
        assert identity_dataset.attrs["model"] == model_str

    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_write_identity_data_with_default_parameters(
        self, mock_h5py_file, mock_adjust_pose_version
    ):
        """Test writing identity data with default config and model strings."""
        # Arrange
        pose_file = "test_pose.h5"
        pose_data_shape = (50, 2, 12, 2)
        embeddings = np.random.rand(50, 2, 64).astype(np.float32)

        existing_datasets = ["poseest/points"]
        mock_context = create_mock_h5_context(existing_datasets, pose_data_shape)
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Act
        write_identity_data(pose_file, embeddings)

        # Assert
        # Should set empty string attributes by default
        identity_info = mock_context.created_datasets["poseest/identity_embeds"]
        identity_dataset = identity_info["dataset"]
        assert identity_dataset.attrs["config"] == ""
        assert identity_dataset.attrs["model"] == ""

    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_overwrite_existing_identity_dataset(
        self, mock_h5py_file, mock_adjust_pose_version
    ):
        """Test that existing identity dataset is properly overwritten."""
        # Arrange
        pose_file = "test_pose.h5"
        pose_data_shape = (75, 4, 12, 2)
        embeddings = np.random.rand(75, 4, 256).astype(np.float32)
        config_str = "new_config"
        model_str = "new_model"

        # Mock existing identity dataset
        existing_datasets = ["poseest/points", "poseest/identity_embeds"]
        mock_context = create_mock_h5_context(existing_datasets, pose_data_shape)
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Act
        write_identity_data(pose_file, embeddings, config_str, model_str)

        # Assert
        # Should delete existing dataset before creating new one
        assert "poseest/identity_embeds" in mock_context.deleted_datasets

        # Should create new dataset
        assert "poseest/identity_embeds" in mock_context.created_datasets

    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_single_animal_identity_data(
        self, mock_h5py_file, mock_adjust_pose_version
    ):
        """Test writing identity data for single animal."""
        # Arrange
        pose_file = "test_pose.h5"
        pose_data_shape = (200, 1, 12, 2)  # Single animal
        embeddings = np.random.rand(200, 1, 512).astype(np.float32)
        config_str = "single_animal_config"
        model_str = "single_animal_model"

        existing_datasets = ["poseest/points"]
        mock_context = create_mock_h5_context(existing_datasets, pose_data_shape)
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Act
        write_identity_data(pose_file, embeddings, config_str, model_str)

        # Assert
        # Should successfully create dataset with correct data
        identity_info = mock_context.created_datasets["poseest/identity_embeds"]
        np.testing.assert_array_equal(
            identity_info["data"], embeddings.astype(np.float32)
        )

        # Verify attributes
        identity_dataset = identity_info["dataset"]
        assert identity_dataset.attrs["config"] == config_str
        assert identity_dataset.attrs["model"] == model_str

    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_multiple_animals_identity_data(
        self, mock_h5py_file, mock_adjust_pose_version
    ):
        """Test writing identity data for multiple animals."""
        # Arrange
        pose_file = "test_pose.h5"
        pose_data_shape = (300, 5, 12, 2)  # 5 animals
        embeddings = np.random.rand(300, 5, 256).astype(np.float32)

        existing_datasets = ["poseest/points"]
        mock_context = create_mock_h5_context(existing_datasets, pose_data_shape)
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Act
        write_identity_data(pose_file, embeddings)

        # Assert
        # Should successfully handle multiple animals
        assert "poseest/identity_embeds" in mock_context.created_datasets

        identity_info = mock_context.created_datasets["poseest/identity_embeds"]
        assert identity_info["data"].shape == (300, 5, 256)
        assert identity_info["data"].dtype == np.float32


class TestWriteIdentityDataErrorHandling:
    """Test error handling for write_identity_data."""

    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_embedding_shape_mismatch_raises_exception(
        self, mock_h5py_file, mock_adjust_pose_version
    ):
        """Test that mismatched embedding shape raises InvalidPoseFileException."""
        # Arrange
        pose_file = "test_pose.h5"
        pose_data_shape = (100, 3, 12, 2)  # [100 frames, 3 animals]
        embeddings = np.random.rand(100, 2, 128).astype(
            np.float32
        )  # Wrong animal count

        existing_datasets = ["poseest/points"]
        mock_context = create_mock_h5_context(existing_datasets, pose_data_shape)
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Act & Assert
        with pytest.raises(
            InvalidPoseFileException,
            match="Keypoint data does not match embedding data shape",
        ):
            write_identity_data(pose_file, embeddings)

        # Should still call adjust_pose_version before validation
        mock_adjust_pose_version.assert_called_once_with(pose_file, 4)

    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_frame_count_mismatch_raises_exception(
        self, mock_h5py_file, mock_adjust_pose_version
    ):
        """Test that mismatched frame count raises InvalidPoseFileException."""
        # Arrange
        pose_file = "test_pose.h5"
        pose_data_shape = (100, 2, 12, 2)  # [100 frames, 2 animals]
        embeddings = np.random.rand(80, 2, 128).astype(np.float32)  # Wrong frame count

        existing_datasets = ["poseest/points"]
        mock_context = create_mock_h5_context(existing_datasets, pose_data_shape)
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Act & Assert
        with pytest.raises(
            InvalidPoseFileException,
            match="Keypoint data does not match embedding data shape",
        ):
            write_identity_data(pose_file, embeddings)

    @pytest.mark.parametrize(
        "pose_shape,embedding_shape,expected_error",
        [
            (
                (100, 3, 12, 2),  # pose_data[:2] = (100, 3)
                (100, 2, 128),  # wrong animals
                "Keypoint data does not match embedding data shape",
            ),
            (
                (100, 3, 12, 2),  # pose_data[:2] = (100, 3)
                (90, 3, 128),  # wrong frames
                "Keypoint data does not match embedding data shape",
            ),
            (
                (100, 3, 12, 2),  # pose_data[:2] = (100, 3)
                (80, 2, 128),  # wrong both
                "Keypoint data does not match embedding data shape",
            ),
            (
                (50, 1, 12, 2),  # pose_data[:2] = (50, 1)
                (60, 2, 256),  # wrong both
                "Keypoint data does not match embedding data shape",
            ),
        ],
        ids=[
            "animals_mismatch",
            "frames_mismatch",
            "both_mismatch",
            "single_to_multi_mismatch",
        ],
    )
    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_various_shape_mismatches(
        self,
        mock_h5py_file,
        mock_adjust_pose_version,
        pose_shape,
        embedding_shape,
        expected_error,
    ):
        """Test various combinations of shape mismatches."""
        # Arrange
        pose_file = "test_pose.h5"
        embeddings = np.random.rand(*embedding_shape).astype(np.float32)

        existing_datasets = ["poseest/points"]
        mock_context = create_mock_h5_context(existing_datasets, pose_shape)
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Act & Assert
        with pytest.raises(InvalidPoseFileException, match=expected_error):
            write_identity_data(pose_file, embeddings)


class TestWriteIdentityDataDataTypes:
    """Test data type handling for write_identity_data."""

    @pytest.mark.parametrize(
        "input_dtype,expected_output_dtype",
        [
            (np.float16, np.float32),
            (np.float64, np.float32),
            (np.int32, np.float32),
            (np.int64, np.float32),
            (np.uint32, np.float32),
        ],
        ids=["float16", "float64", "int32", "int64", "uint32"],
    )
    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_data_type_conversion_embeddings(
        self,
        mock_h5py_file,
        mock_adjust_pose_version,
        input_dtype,
        expected_output_dtype,
    ):
        """Test that embeddings are converted to float32."""
        # Arrange
        pose_file = "test_pose.h5"
        pose_data_shape = (50, 2, 12, 2)
        embeddings = np.random.rand(50, 2, 128).astype(input_dtype)

        existing_datasets = ["poseest/points"]
        mock_context = create_mock_h5_context(existing_datasets, pose_data_shape)
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Act
        write_identity_data(pose_file, embeddings)

        # Assert
        identity_info = mock_context.created_datasets["poseest/identity_embeds"]
        assert identity_info["data"].dtype == expected_output_dtype

    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_negative_values_handled_correctly(
        self, mock_h5py_file, mock_adjust_pose_version
    ):
        """Test handling of negative values in embedding data."""
        # Arrange
        pose_file = "test_pose.h5"
        pose_data_shape = (3, 2, 12, 2)
        # Include negative values which should be preserved
        embeddings = np.array(
            [
                [[-1.5, 0.5, 2.3], [1.0, -2.1, 0.8]],
                [[0.0, -0.5, 1.2], [-1.8, 3.4, -0.2]],
                [[2.1, -3.0, 0.7], [0.9, 1.5, -2.5]],
            ],
            dtype=np.float64,
        )

        existing_datasets = ["poseest/points"]
        mock_context = create_mock_h5_context(existing_datasets, pose_data_shape)
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Act
        write_identity_data(pose_file, embeddings)

        # Assert
        identity_info = mock_context.created_datasets["poseest/identity_embeds"]

        # Verify that negative values are preserved
        expected_embeddings = embeddings.astype(np.float32)
        np.testing.assert_array_equal(identity_info["data"], expected_embeddings)

    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_extreme_values_handled_correctly(
        self, mock_h5py_file, mock_adjust_pose_version
    ):
        """Test handling of extreme float values."""
        # Arrange
        pose_file = "test_pose.h5"
        pose_data_shape = (2, 1, 12, 2)
        # Use extreme values
        max_float32 = np.finfo(np.float32).max
        min_float32 = np.finfo(np.float32).min
        embeddings = np.array(
            [[[max_float32, min_float32, 0.0]], [[np.inf, -np.inf, np.nan]]],
            dtype=np.float64,
        )

        existing_datasets = ["poseest/points"]
        mock_context = create_mock_h5_context(existing_datasets, pose_data_shape)
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Act
        write_identity_data(pose_file, embeddings)

        # Assert
        identity_info = mock_context.created_datasets["poseest/identity_embeds"]
        assert identity_info["data"].dtype == np.float32

        # Check that conversion was applied
        expected_embeddings = embeddings.astype(np.float32)
        np.testing.assert_array_equal(identity_info["data"], expected_embeddings)


class TestWriteIdentityDataVersionHandling:
    """Test version promotion handling for write_identity_data."""

    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_adjust_pose_version_called_before_writing(
        self, mock_h5py_file, mock_adjust_pose_version
    ):
        """Test that adjust_pose_version is called before writing data."""
        # Arrange
        pose_file = "test_pose.h5"
        pose_data_shape = (30, 2, 12, 2)
        embeddings = np.random.rand(30, 2, 64).astype(np.float32)

        existing_datasets = ["poseest/points"]
        mock_context = create_mock_h5_context(existing_datasets, pose_data_shape)
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Act
        write_identity_data(pose_file, embeddings)

        # Assert
        # Should call adjust_pose_version with version 4
        mock_adjust_pose_version.assert_called_once_with(pose_file, 4)

        # Verify adjust_pose_version was called before h5py.File
        assert mock_adjust_pose_version.call_count == 1
        assert mock_h5py_file.call_count == 1

    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_version_promotion_failure_prevents_writing(
        self, mock_h5py_file, mock_adjust_pose_version
    ):
        """Test that if version promotion fails, writing doesn't proceed."""
        # Arrange
        pose_file = "test_pose.h5"
        embeddings = np.random.rand(50, 3, 128).astype(np.float32)

        # Mock adjust_pose_version to raise an exception
        mock_adjust_pose_version.side_effect = Exception("Version promotion failed")

        # Act & Assert
        with pytest.raises(Exception, match="Version promotion failed"):
            write_identity_data(pose_file, embeddings)

        # Should not attempt to open the file if version promotion fails
        mock_h5py_file.assert_not_called()


class TestWriteIdentityDataEdgeCases:
    """Test edge cases for write_identity_data."""

    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_empty_data_arrays(self, mock_h5py_file, mock_adjust_pose_version):
        """Test handling of empty data arrays."""
        # Arrange
        pose_file = "test_pose.h5"
        pose_data_shape = (0, 0, 12, 2)  # Empty frame and animal dimensions
        embeddings = np.array([], dtype=np.float32).reshape(0, 0, 128)

        existing_datasets = ["poseest/points"]
        mock_context = create_mock_h5_context(existing_datasets, pose_data_shape)
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Act
        write_identity_data(pose_file, embeddings)

        # Assert
        # Should successfully create dataset even with empty data
        assert "poseest/identity_embeds" in mock_context.created_datasets

        identity_info = mock_context.created_datasets["poseest/identity_embeds"]
        assert identity_info["data"].shape == (0, 0, 128)
        assert identity_info["data"].dtype == np.float32

    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_single_frame_data(self, mock_h5py_file, mock_adjust_pose_version):
        """Test handling of single frame data."""
        # Arrange
        pose_file = "test_pose.h5"
        pose_data_shape = (1, 3, 12, 2)  # Single frame
        embeddings = np.random.rand(1, 3, 256).astype(np.float32)
        config_str = "single_frame_config"
        model_str = "single_frame_model"

        existing_datasets = ["poseest/points"]
        mock_context = create_mock_h5_context(existing_datasets, pose_data_shape)
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Act
        write_identity_data(pose_file, embeddings, config_str, model_str)

        # Assert
        identity_info = mock_context.created_datasets["poseest/identity_embeds"]
        np.testing.assert_array_equal(identity_info["data"], embeddings)

        # Verify attributes are set correctly
        identity_dataset = identity_info["dataset"]
        assert identity_dataset.attrs["config"] == config_str
        assert identity_dataset.attrs["model"] == model_str

    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_zero_embedding_dimension(self, mock_h5py_file, mock_adjust_pose_version):
        """Test handling of zero embedding dimension."""
        # Arrange
        pose_file = "test_pose.h5"
        pose_data_shape = (50, 2, 12, 2)
        embeddings = np.array([], dtype=np.float32).reshape(50, 2, 0)  # Zero embed dim

        existing_datasets = ["poseest/points"]
        mock_context = create_mock_h5_context(existing_datasets, pose_data_shape)
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Act
        write_identity_data(pose_file, embeddings)

        # Assert
        identity_info = mock_context.created_datasets["poseest/identity_embeds"]
        assert identity_info["data"].shape == (50, 2, 0)
        assert identity_info["data"].dtype == np.float32

    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_large_embedding_dimension(self, mock_h5py_file, mock_adjust_pose_version):
        """Test handling of large embedding dimension."""
        # Arrange
        pose_file = "test_pose.h5"
        pose_data_shape = (10, 1, 12, 2)
        embeddings = np.random.rand(10, 1, 2048).astype(np.float32)  # Large embed dim

        existing_datasets = ["poseest/points"]
        mock_context = create_mock_h5_context(existing_datasets, pose_data_shape)
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Act
        write_identity_data(pose_file, embeddings)

        # Assert
        identity_info = mock_context.created_datasets["poseest/identity_embeds"]
        assert identity_info["data"].shape == (10, 1, 2048)
        np.testing.assert_array_equal(identity_info["data"], embeddings)

    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_string_attributes_with_special_characters(
        self, mock_h5py_file, mock_adjust_pose_version
    ):
        """Test setting attributes with special characters."""
        # Arrange
        pose_file = "test_pose.h5"
        pose_data_shape = (20, 1, 12, 2)
        embeddings = np.random.rand(20, 1, 64).astype(np.float32)
        config_str = "config/with/slashes_and-dashes & symbols"
        model_str = "model:checkpoint@v1.0 (final)"

        existing_datasets = ["poseest/points"]
        mock_context = create_mock_h5_context(existing_datasets, pose_data_shape)
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Act
        write_identity_data(pose_file, embeddings, config_str, model_str)

        # Assert
        identity_info = mock_context.created_datasets["poseest/identity_embeds"]
        identity_dataset = identity_info["dataset"]
        assert identity_dataset.attrs["config"] == config_str
        assert identity_dataset.attrs["model"] == model_str


class TestWriteIdentityDataIntegration:
    """Integration-style tests for write_identity_data."""

    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_complete_workflow_with_realistic_data(
        self, mock_h5py_file, mock_adjust_pose_version
    ):
        """Test complete workflow with realistic identity embedding data."""
        # Arrange
        pose_file = "realistic_identity.h5"
        num_frames = 500
        num_animals = 3
        embed_dim = 256
        pose_data_shape = (num_frames, num_animals, 12, 2)

        # Create realistic embedding data with some variability
        embeddings = np.random.randn(num_frames, num_animals, embed_dim).astype(
            np.float32
        )
        # Normalize embeddings as would typically be done in real identity models
        embeddings = embeddings / np.linalg.norm(
            embeddings, axis=-1, keepdims=True
        ).clip(min=1e-8)

        config_str = "resnet18_identity_model_v2.yaml"
        model_str = "identity_checkpoint_epoch_100.pth"

        existing_datasets = ["poseest/points"]
        mock_context = create_mock_h5_context(existing_datasets, pose_data_shape)
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Act
        write_identity_data(pose_file, embeddings, config_str, model_str)

        # Assert
        # Verify version promotion was called
        mock_adjust_pose_version.assert_called_once_with(pose_file, 4)

        # Verify dataset was created correctly
        assert "poseest/identity_embeds" in mock_context.created_datasets
        identity_info = mock_context.created_datasets["poseest/identity_embeds"]

        # Verify data integrity
        np.testing.assert_array_equal(
            identity_info["data"], embeddings.astype(np.float32)
        )

        # Verify data properties
        assert identity_info["data"].dtype == np.float32
        assert identity_info["data"].shape == (num_frames, num_animals, embed_dim)

        # Verify attributes
        identity_dataset = identity_info["dataset"]
        assert identity_dataset.attrs["config"] == config_str
        assert identity_dataset.attrs["model"] == model_str

    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_workflow_with_dataset_replacement(
        self, mock_h5py_file, mock_adjust_pose_version
    ):
        """Test workflow where existing identity dataset is replaced."""
        # Arrange
        pose_file = "test_pose.h5"
        pose_data_shape = (100, 2, 12, 2)
        embeddings = np.random.rand(100, 2, 128).astype(np.float32)
        config_str = "updated_config"
        model_str = "updated_model"

        # Mock existing identity dataset that will be replaced
        existing_datasets = ["poseest/points", "poseest/identity_embeds"]
        mock_context = create_mock_h5_context(existing_datasets, pose_data_shape)
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Act
        write_identity_data(pose_file, embeddings, config_str, model_str)

        # Assert
        # Should delete existing dataset
        assert "poseest/identity_embeds" in mock_context.deleted_datasets

        # Should create new dataset with correct data
        assert "poseest/identity_embeds" in mock_context.created_datasets
        identity_info = mock_context.created_datasets["poseest/identity_embeds"]

        np.testing.assert_array_equal(identity_info["data"], embeddings)

        # Verify new attributes
        identity_dataset = identity_info["dataset"]
        assert identity_dataset.attrs["config"] == config_str
        assert identity_dataset.attrs["model"] == model_str

    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_workflow_with_version_promotion_and_validation(
        self, mock_h5py_file, mock_adjust_pose_version
    ):
        """Test complete workflow ensuring version promotion happens before validation."""
        # Arrange
        pose_file = "test_pose.h5"
        pose_data_shape = (80, 4, 12, 2)
        embeddings = np.random.rand(80, 4, 512).astype(np.float64)

        existing_datasets = ["poseest/points"]
        mock_context = create_mock_h5_context(existing_datasets, pose_data_shape)
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Act
        write_identity_data(pose_file, embeddings)

        # Assert
        # Verify call order: adjust_pose_version should be called first
        mock_adjust_pose_version.assert_called_once_with(pose_file, 4)

        # File should be opened after version promotion
        mock_h5py_file.assert_called_once_with(pose_file, "a")

        # Data should be written with correct type conversion
        identity_info = mock_context.created_datasets["poseest/identity_embeds"]
        assert identity_info["data"].dtype == np.float32
        np.testing.assert_array_equal(
            identity_info["data"], embeddings.astype(np.float32)
        )
