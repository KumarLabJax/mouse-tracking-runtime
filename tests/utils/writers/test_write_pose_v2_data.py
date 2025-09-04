"""Comprehensive unit tests for the write_pose_v2_data function."""

from unittest.mock import patch

import numpy as np
import pytest

from mouse_tracking.core.exceptions import InvalidPoseFileException
from mouse_tracking.utils.writers import write_pose_v2_data

from .mock_hdf5 import create_mock_h5_context


class TestWritePoseV2DataBasicFunctionality:
    """Test basic functionality of write_pose_v2_data."""

    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_write_single_animal_pose_data_success(
        self, mock_h5py_file, mock_adjust_pose_version
    ):
        """Test successful writing of single animal pose data."""
        # Arrange
        pose_file = "test_pose.h5"
        pose_matrix = np.random.rand(100, 12, 2).astype(np.float32)
        confidence_matrix = np.random.rand(100, 12).astype(np.float32)
        config_str = "test_config"
        model_str = "test_model"

        mock_context = create_mock_h5_context()
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Act
        write_pose_v2_data(
            pose_file, pose_matrix, confidence_matrix, config_str, model_str
        )

        # Assert
        # Should open file in append mode
        mock_h5py_file.assert_called_once_with(pose_file, "a")

        # Should create pose points dataset
        assert "poseest/points" in mock_context.created_datasets
        points_info = mock_context.created_datasets["poseest/points"]
        np.testing.assert_array_equal(
            points_info["data"], pose_matrix.astype(np.uint16)
        )

        # Should create confidence dataset
        assert "poseest/confidence" in mock_context.created_datasets
        conf_info = mock_context.created_datasets["poseest/confidence"]
        np.testing.assert_array_equal(
            conf_info["data"], confidence_matrix.astype(np.float32)
        )

        # Should set attributes on points dataset
        points_dataset = points_info["dataset"]
        assert points_dataset.attrs["config"] == config_str
        assert points_dataset.attrs["model"] == model_str

        # Should call adjust_pose_version for single animal (version 2)
        mock_adjust_pose_version.assert_called_once_with(pose_file, 2)

    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_write_multi_animal_pose_data_success(
        self, mock_h5py_file, mock_adjust_pose_version
    ):
        """Test successful writing of multi-animal pose data."""
        # Arrange
        pose_file = "test_pose.h5"
        pose_matrix = np.random.rand(100, 3, 12, 2).astype(np.float32)  # 3 animals
        confidence_matrix = np.random.rand(100, 3, 12).astype(np.float32)
        config_str = "multi_config"
        model_str = "multi_model"

        mock_context = create_mock_h5_context()
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Act
        write_pose_v2_data(
            pose_file, pose_matrix, confidence_matrix, config_str, model_str
        )

        # Assert
        # Should create datasets with correct data types
        points_info = mock_context.created_datasets["poseest/points"]
        np.testing.assert_array_equal(
            points_info["data"], pose_matrix.astype(np.uint16)
        )

        conf_info = mock_context.created_datasets["poseest/confidence"]
        np.testing.assert_array_equal(
            conf_info["data"], confidence_matrix.astype(np.float32)
        )

        # Should call adjust_pose_version for multi-animal (version 3, no promotion)
        mock_adjust_pose_version.assert_called_once_with(pose_file, 3, False)

    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_write_pose_data_with_default_parameters(
        self, mock_h5py_file, mock_adjust_pose_version
    ):
        """Test writing pose data with default config and model strings."""
        # Arrange
        pose_file = "test_pose.h5"
        pose_matrix = np.random.rand(50, 12, 2).astype(np.float32)
        confidence_matrix = np.random.rand(50, 12).astype(np.float32)

        mock_context = create_mock_h5_context()
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Act
        write_pose_v2_data(pose_file, pose_matrix, confidence_matrix)

        # Assert
        # Should set empty string attributes by default
        points_info = mock_context.created_datasets["poseest/points"]
        points_dataset = points_info["dataset"]
        assert points_dataset.attrs["config"] == ""
        assert points_dataset.attrs["model"] == ""

    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_overwrite_existing_datasets(
        self, mock_h5py_file, mock_adjust_pose_version
    ):
        """Test that existing datasets are properly overwritten."""
        # Arrange
        pose_file = "test_pose.h5"
        pose_matrix = np.random.rand(75, 12, 2).astype(np.float32)
        confidence_matrix = np.random.rand(75, 12).astype(np.float32)

        # Mock context with existing datasets
        existing_datasets = ["poseest/points", "poseest/confidence"]
        mock_context = create_mock_h5_context(existing_datasets)
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Track deletions
        deleted_datasets = []

        def track_delitem(self, key):
            deleted_datasets.append(key)

        mock_context.__delitem__ = track_delitem

        # Act
        write_pose_v2_data(pose_file, pose_matrix, confidence_matrix)

        # Assert
        # Should delete existing datasets
        assert "poseest/points" in deleted_datasets
        assert "poseest/confidence" in deleted_datasets

        # Should create new datasets
        assert "poseest/points" in mock_context.created_datasets
        assert "poseest/confidence" in mock_context.created_datasets


class TestWritePoseV2DataErrorHandling:
    """Test error handling in write_pose_v2_data."""

    def test_mismatched_frame_counts_raises_exception(self):
        """Test that mismatched frame counts raise InvalidPoseFileException."""
        # Arrange
        pose_file = "test_pose.h5"
        pose_matrix = np.random.rand(100, 12, 2).astype(np.float32)
        confidence_matrix = np.random.rand(90, 12).astype(
            np.float32
        )  # Different frame count

        # Act & Assert
        with pytest.raises(
            InvalidPoseFileException,
            match="Pose data does not match confidence data. Pose shape: 100, Confidence shape: 90",
        ):
            write_pose_v2_data(pose_file, pose_matrix, confidence_matrix)

    def test_mixed_single_multi_dimensions_raises_exception(self):
        """Test that mixed single/multi animal dimensions raise InvalidPoseFileException."""
        # Arrange
        pose_file = "test_pose.h5"
        pose_matrix = np.random.rand(100, 12, 2).astype(
            np.float32
        )  # Single animal format
        confidence_matrix = np.random.rand(100, 3, 12).astype(
            np.float32
        )  # Multi animal format

        # Act & Assert
        with pytest.raises(
            InvalidPoseFileException,
            match="Pose dimensions are mixed between single and multi animal formats. Pose dim: 3, Confidence dim: 3",
        ):
            write_pose_v2_data(pose_file, pose_matrix, confidence_matrix)

    def test_invalid_pose_dimensions_raises_exception(self):
        """Test that invalid pose dimensions raise InvalidPoseFileException."""
        # Arrange
        pose_file = "test_pose.h5"
        pose_matrix = np.random.rand(100, 12).astype(
            np.float32
        )  # Missing coordinate dimension
        confidence_matrix = np.random.rand(100, 12).astype(np.float32)

        # Act & Assert
        with pytest.raises(
            InvalidPoseFileException,
            match="Pose dimensions are mixed between single and multi animal formats. Pose dim: 2, Confidence dim: 2",
        ):
            write_pose_v2_data(pose_file, pose_matrix, confidence_matrix)

    @pytest.mark.parametrize(
        "pose_shape,conf_shape,expected_error",
        [
            (
                (100, 12),
                (100, 12),
                "Pose dimensions are mixed between single and multi animal formats",
            ),
            (
                (100, 2, 12, 2),
                (100, 12),
                "Pose dimensions are mixed between single and multi animal formats",
            ),
            ((50, 12, 2), (60, 12), "Pose data does not match confidence data"),
            (
                (100, 3, 12),
                (100, 3, 12),
                "Pose dimensions are mixed between single and multi animal formats",
            ),
        ],
        ids=[
            "both_2d",
            "pose_4d_conf_2d",
            "frame_mismatch",
            "both_3d_no_coords",
        ],
    )
    def test_various_dimension_mismatches(self, pose_shape, conf_shape, expected_error):
        """Test various dimension mismatch scenarios."""
        # Arrange
        pose_file = "test_pose.h5"
        pose_matrix = np.random.rand(*pose_shape).astype(np.float32)
        confidence_matrix = np.random.rand(*conf_shape).astype(np.float32)

        # Act & Assert
        with pytest.raises(InvalidPoseFileException, match=expected_error):
            write_pose_v2_data(pose_file, pose_matrix, confidence_matrix)


class TestWritePoseV2DataDataTypes:
    """Test data type handling in write_pose_v2_data."""

    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_data_type_conversion(self, mock_h5py_file, mock_adjust_pose_version):
        """Test that data is properly converted to required types."""
        # Arrange
        pose_file = "test_pose.h5"
        # Use different input data types
        pose_matrix = np.random.rand(50, 12, 2).astype(np.float64)
        confidence_matrix = np.random.rand(50, 12).astype(np.float64)

        mock_context = create_mock_h5_context()
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Act
        write_pose_v2_data(pose_file, pose_matrix, confidence_matrix)

        # Assert
        # Should convert pose data to uint16
        points_info = mock_context.created_datasets["poseest/points"]
        assert points_info["data"].dtype == np.uint16

        # Should convert confidence data to float32
        conf_info = mock_context.created_datasets["poseest/confidence"]
        assert conf_info["data"].dtype == np.float32

    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    @patch("mouse_tracking.utils.writers.h5py.File")
    @pytest.mark.parametrize(
        "input_dtype,expected_output_dtype",
        [
            (np.int32, np.uint16),
            (np.float32, np.uint16),
            (np.float64, np.uint16),
            (np.int64, np.uint16),
        ],
        ids=["int32", "float32", "float64", "int64"],
    )
    def test_pose_data_type_conversions(
        self,
        mock_h5py_file,
        mock_adjust_pose_version,
        input_dtype,
        expected_output_dtype,
    ):
        """Test pose data type conversions from various input types."""
        # Arrange
        pose_file = "test_pose.h5"
        pose_matrix = np.random.rand(30, 12, 2).astype(input_dtype)
        confidence_matrix = np.random.rand(30, 12).astype(np.float32)

        mock_context = create_mock_h5_context()
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Act
        write_pose_v2_data(pose_file, pose_matrix, confidence_matrix)

        # Assert
        points_info = mock_context.created_datasets["poseest/points"]
        assert points_info["data"].dtype == expected_output_dtype

    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    @patch("mouse_tracking.utils.writers.h5py.File")
    @pytest.mark.parametrize(
        "input_dtype,expected_output_dtype",
        [
            (np.float16, np.float32),
            (np.float64, np.float32),
            (np.int32, np.float32),
        ],
        ids=["float16", "float64", "int32"],
    )
    def test_confidence_data_type_conversions(
        self,
        mock_h5py_file,
        mock_adjust_pose_version,
        input_dtype,
        expected_output_dtype,
    ):
        """Test confidence data type conversions from various input types."""
        # Arrange
        pose_file = "test_pose.h5"
        pose_matrix = np.random.rand(30, 12, 2).astype(np.float32)
        confidence_matrix = np.random.rand(30, 12).astype(input_dtype)

        mock_context = create_mock_h5_context()
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Act
        write_pose_v2_data(pose_file, pose_matrix, confidence_matrix)

        # Assert
        conf_info = mock_context.created_datasets["poseest/confidence"]
        assert conf_info["data"].dtype == expected_output_dtype


class TestWritePoseV2DataVersionHandling:
    """Test version handling logic in write_pose_v2_data."""

    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_single_animal_calls_version_2(
        self, mock_h5py_file, mock_adjust_pose_version
    ):
        """Test that single animal data calls adjust_pose_version with version 2."""
        # Arrange
        pose_file = "test_pose.h5"
        pose_matrix = np.random.rand(50, 12, 2).astype(np.float32)
        confidence_matrix = np.random.rand(50, 12).astype(np.float32)

        mock_context = create_mock_h5_context()
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Act
        write_pose_v2_data(pose_file, pose_matrix, confidence_matrix)

        # Assert
        mock_adjust_pose_version.assert_called_once_with(pose_file, 2)

    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_multi_animal_calls_version_3_no_promotion(
        self, mock_h5py_file, mock_adjust_pose_version
    ):
        """Test that multi-animal data calls adjust_pose_version with version 3 and no promotion."""
        # Arrange
        pose_file = "test_pose.h5"
        pose_matrix = np.random.rand(50, 2, 12, 2).astype(np.float32)
        confidence_matrix = np.random.rand(50, 2, 12).astype(np.float32)

        mock_context = create_mock_h5_context()
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Act
        write_pose_v2_data(pose_file, pose_matrix, confidence_matrix)

        # Assert
        mock_adjust_pose_version.assert_called_once_with(pose_file, 3, False)

    @pytest.mark.parametrize(
        "pose_shape,conf_shape,expected_version,expected_promote",
        [
            ((100, 12, 2), (100, 12), 2, True),  # Single animal
            ((100, 1, 12, 2), (100, 1, 12), 3, False),  # Multi-animal (1 animal)
            ((100, 3, 12, 2), (100, 3, 12), 3, False),  # Multi-animal (3 animals)
            ((50, 5, 12, 2), (50, 5, 12), 3, False),  # Multi-animal (5 animals)
        ],
        ids=[
            "single_animal",
            "multi_animal_1",
            "multi_animal_3",
            "multi_animal_5",
        ],
    )
    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_version_handling_matrix(
        self,
        mock_h5py_file,
        mock_adjust_pose_version,
        pose_shape,
        conf_shape,
        expected_version,
        expected_promote,
    ):
        """Test version handling for various input shapes."""
        # Arrange
        pose_file = "test_pose.h5"
        pose_matrix = np.random.rand(*pose_shape).astype(np.float32)
        confidence_matrix = np.random.rand(*conf_shape).astype(np.float32)

        mock_context = create_mock_h5_context()
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Act
        write_pose_v2_data(pose_file, pose_matrix, confidence_matrix)

        # Assert
        if expected_promote:
            mock_adjust_pose_version.assert_called_once_with(
                pose_file, expected_version
            )
        else:
            mock_adjust_pose_version.assert_called_once_with(
                pose_file, expected_version, False
            )


class TestWritePoseV2DataEdgeCases:
    """Test edge cases and boundary conditions of write_pose_v2_data."""

    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_empty_data_arrays(self, mock_h5py_file, mock_adjust_pose_version):
        """Test handling of empty data arrays."""
        # Arrange
        pose_file = "test_pose.h5"
        pose_matrix = np.empty((0, 12, 2), dtype=np.float32)
        confidence_matrix = np.empty((0, 12), dtype=np.float32)

        mock_context = create_mock_h5_context()
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Act
        write_pose_v2_data(pose_file, pose_matrix, confidence_matrix)

        # Assert
        # Should still create datasets even with empty data
        assert "poseest/points" in mock_context.created_datasets
        assert "poseest/confidence" in mock_context.created_datasets
        mock_adjust_pose_version.assert_called_once_with(pose_file, 2)

    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_single_frame_data(self, mock_h5py_file, mock_adjust_pose_version):
        """Test handling of single frame data."""
        # Arrange
        pose_file = "test_pose.h5"
        pose_matrix = np.random.rand(1, 12, 2).astype(np.float32)
        confidence_matrix = np.random.rand(1, 12).astype(np.float32)

        mock_context = create_mock_h5_context()
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Act
        write_pose_v2_data(pose_file, pose_matrix, confidence_matrix)

        # Assert
        points_info = mock_context.created_datasets["poseest/points"]
        assert points_info["data"].shape == (1, 12, 2)
        mock_adjust_pose_version.assert_called_once_with(pose_file, 2)

    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_string_attributes_with_special_characters(
        self, mock_h5py_file, mock_adjust_pose_version
    ):
        """Test handling of string attributes with special characters."""
        # Arrange
        pose_file = "test_pose.h5"
        pose_matrix = np.random.rand(10, 12, 2).astype(np.float32)
        confidence_matrix = np.random.rand(10, 12).astype(np.float32)
        config_str = "config with spaces & symbols: αβγ"
        model_str = "model_path/with/slashes\\and\\backslashes"

        mock_context = create_mock_h5_context()
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Act
        write_pose_v2_data(
            pose_file, pose_matrix, confidence_matrix, config_str, model_str
        )

        # Assert
        points_dataset = mock_context.created_datasets["poseest/points"]["dataset"]
        assert points_dataset.attrs["config"] == config_str
        assert points_dataset.attrs["model"] == model_str


class TestWritePoseV2DataIntegration:
    """Test integration scenarios for write_pose_v2_data."""

    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_complete_workflow_single_animal(
        self, mock_h5py_file, mock_adjust_pose_version
    ):
        """Test complete workflow for single animal data writing."""
        # Arrange
        pose_file = "/path/to/test_pose.h5"
        pose_matrix = np.random.rand(1000, 12, 2).astype(np.float32)
        confidence_matrix = np.random.rand(1000, 12).astype(np.float32)
        config_str = "hrnet_config_v1.yaml"
        model_str = "model_checkpoint_epoch_100.pth"

        mock_context = create_mock_h5_context(["poseest/points"])  # Existing dataset
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        deleted_datasets = []

        def track_delitem(self, key):
            deleted_datasets.append(key)

        mock_context.__delitem__ = track_delitem

        # Act
        write_pose_v2_data(
            pose_file, pose_matrix, confidence_matrix, config_str, model_str
        )

        # Assert
        # Should open file correctly
        mock_h5py_file.assert_called_once_with(pose_file, "a")

        # Should delete existing dataset
        assert "poseest/points" in deleted_datasets

        # Should create both datasets with correct data
        assert "poseest/points" in mock_context.created_datasets
        assert "poseest/confidence" in mock_context.created_datasets

        # Should set attributes correctly
        points_dataset = mock_context.created_datasets["poseest/points"]["dataset"]
        assert points_dataset.attrs["config"] == config_str
        assert points_dataset.attrs["model"] == model_str

        # Should call version adjustment
        mock_adjust_pose_version.assert_called_once_with(pose_file, 2)

    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_complete_workflow_multi_animal(
        self, mock_h5py_file, mock_adjust_pose_version
    ):
        """Test complete workflow for multi-animal data writing."""
        # Arrange
        pose_file = "/path/to/multi_pose.h5"
        num_animals = 4
        pose_matrix = np.random.rand(500, num_animals, 12, 2).astype(np.float32)
        confidence_matrix = np.random.rand(500, num_animals, 12).astype(np.float32)
        config_str = "multi_animal_config.yaml"
        model_str = "multi_animal_model.pth"

        existing_datasets = ["poseest/points", "poseest/confidence"]
        mock_context = create_mock_h5_context(existing_datasets)
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        deleted_datasets = []

        def track_delitem(self, key):
            deleted_datasets.append(key)

        mock_context.__delitem__ = track_delitem

        # Act
        write_pose_v2_data(
            pose_file, pose_matrix, confidence_matrix, config_str, model_str
        )

        # Assert
        # Should delete both existing datasets
        assert "poseest/points" in deleted_datasets
        assert "poseest/confidence" in deleted_datasets

        # Should create datasets with correct data types and shapes
        points_info = mock_context.created_datasets["poseest/points"]
        assert points_info["data"].shape == (500, num_animals, 12, 2)
        assert points_info["data"].dtype == np.uint16

        conf_info = mock_context.created_datasets["poseest/confidence"]
        assert conf_info["data"].shape == (500, num_animals, 12)
        assert conf_info["data"].dtype == np.float32

        # Should call version adjustment for multi-animal
        mock_adjust_pose_version.assert_called_once_with(pose_file, 3, False)
