"""Comprehensive unit tests for the write_seg_data function."""

from unittest.mock import patch

import numpy as np
import pytest

from mouse_tracking.core.exceptions import InvalidPoseFileException
from mouse_tracking.utils.writers import write_seg_data

from .mock_hdf5 import create_mock_h5_context


class TestWriteSegDataBasicFunctionality:
    """Test basic functionality of write_seg_data."""

    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_write_seg_data_success(self, mock_h5py_file, mock_adjust_pose_version):
        """Test successful writing of segmentation data."""
        # Arrange
        pose_file = "test_pose.h5"
        seg_contours_matrix = np.random.randint(
            0, 100, size=(50, 2, 3, 10, 2), dtype=np.int32
        )  # [frame, animals, contours, points, coords]
        seg_external_flags = np.random.randint(
            0, 2, size=(50, 2, 3), dtype=np.int32
        )  # [frame, animals, contours]
        config_str = "test_config"
        model_str = "test_model"

        existing_datasets = []
        mock_context = create_mock_h5_context(existing_datasets)
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Act
        write_seg_data(
            pose_file, seg_contours_matrix, seg_external_flags, config_str, model_str
        )

        # Assert
        # Should open file in append mode
        mock_h5py_file.assert_called_once_with(pose_file, "a")

        # Should create seg_data dataset with compression
        assert "poseest/seg_data" in mock_context.created_datasets
        seg_data_info = mock_context.created_datasets["poseest/seg_data"]
        np.testing.assert_array_equal(seg_data_info["data"], seg_contours_matrix)
        assert seg_data_info["kwargs"]["compression"] == "gzip"
        assert seg_data_info["kwargs"]["compression_opts"] == 9

        # Should create seg_external_flag dataset with compression
        assert "poseest/seg_external_flag" in mock_context.created_datasets
        flag_info = mock_context.created_datasets["poseest/seg_external_flag"]
        np.testing.assert_array_equal(flag_info["data"], seg_external_flags)
        assert flag_info["kwargs"]["compression"] == "gzip"
        assert flag_info["kwargs"]["compression_opts"] == 9

        # Should set attributes on seg_data dataset
        seg_dataset = seg_data_info["dataset"]
        assert seg_dataset.attrs["config"] == config_str
        assert seg_dataset.attrs["model"] == model_str

        # Should call adjust_pose_version by default
        mock_adjust_pose_version.assert_called_once_with(pose_file, 6)

    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_write_seg_data_with_skip_matching(
        self, mock_h5py_file, mock_adjust_pose_version
    ):
        """Test writing segmentation data with skip_matching=True."""
        # Arrange
        pose_file = "test_pose.h5"
        seg_contours_matrix = np.random.randint(
            0, 50, size=(30, 1, 2, 15, 2), dtype=np.int32
        )
        seg_external_flags = np.random.randint(0, 2, size=(30, 1, 2), dtype=np.int32)

        existing_datasets = []
        mock_context = create_mock_h5_context(existing_datasets)
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Act
        write_seg_data(
            pose_file,
            seg_contours_matrix,
            seg_external_flags,
            skip_matching=True,
        )

        # Assert
        # Should create datasets as normal
        assert "poseest/seg_data" in mock_context.created_datasets
        assert "poseest/seg_external_flag" in mock_context.created_datasets

        # Should NOT call adjust_pose_version when skip_matching=True
        mock_adjust_pose_version.assert_not_called()

    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_write_seg_data_with_default_parameters(
        self, mock_h5py_file, mock_adjust_pose_version
    ):
        """Test writing segmentation data with default config and model strings."""
        # Arrange
        pose_file = "test_pose.h5"
        seg_contours_matrix = np.random.randint(
            0, 80, size=(25, 3, 1, 8, 2), dtype=np.int32
        )
        seg_external_flags = np.random.randint(0, 2, size=(25, 3, 1), dtype=np.int32)

        existing_datasets = []
        mock_context = create_mock_h5_context(existing_datasets)
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Act
        write_seg_data(pose_file, seg_contours_matrix, seg_external_flags)

        # Assert
        # Should set empty string attributes by default
        seg_data_info = mock_context.created_datasets["poseest/seg_data"]
        seg_dataset = seg_data_info["dataset"]
        assert seg_dataset.attrs["config"] == ""
        assert seg_dataset.attrs["model"] == ""

    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_overwrite_existing_seg_datasets(
        self, mock_h5py_file, mock_adjust_pose_version
    ):
        """Test that existing segmentation datasets are properly overwritten."""
        # Arrange
        pose_file = "test_pose.h5"
        seg_contours_matrix = np.random.randint(
            0, 60, size=(40, 2, 2, 12, 2), dtype=np.int32
        )
        seg_external_flags = np.random.randint(0, 2, size=(40, 2, 2), dtype=np.int32)
        config_str = "new_config"
        model_str = "new_model"

        # Mock existing datasets
        existing_datasets = ["poseest/seg_data", "poseest/seg_external_flag"]
        mock_context = create_mock_h5_context(existing_datasets)
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Act
        write_seg_data(
            pose_file, seg_contours_matrix, seg_external_flags, config_str, model_str
        )

        # Assert
        # Should delete existing datasets before creating new ones
        assert "poseest/seg_data" in mock_context.deleted_datasets
        assert "poseest/seg_external_flag" in mock_context.deleted_datasets

        # Should create new datasets
        assert "poseest/seg_data" in mock_context.created_datasets
        assert "poseest/seg_external_flag" in mock_context.created_datasets


class TestWriteSegDataErrorHandling:
    """Test error handling for write_seg_data."""

    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_shape_mismatch_raises_exception(
        self, mock_h5py_file, mock_adjust_pose_version
    ):
        """Test that mismatched shapes raise InvalidPoseFileException."""
        # Arrange
        pose_file = "test_pose.h5"
        seg_contours_matrix = np.random.randint(
            0, 50, size=(100, 3, 2, 10, 2), dtype=np.int32
        )  # [100, 3, 2, ...]
        seg_external_flags = np.random.randint(
            0, 2, size=(100, 2, 2), dtype=np.int32
        )  # [100, 2, 2] - wrong animal count

        existing_datasets = []
        mock_context = create_mock_h5_context(existing_datasets)
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Act & Assert
        with pytest.raises(
            InvalidPoseFileException,
            match="Segmentation data shape does not match",
        ):
            write_seg_data(pose_file, seg_contours_matrix, seg_external_flags)

        # Should not call adjust_pose_version when validation fails
        mock_adjust_pose_version.assert_not_called()

    @pytest.mark.parametrize(
        "contours_shape,flags_shape,expected_error",
        [
            (
                (100, 3, 2, 10, 2),  # contours[:3] = (100, 3, 2)
                (100, 2, 2),  # wrong animals
                "Segmentation data shape does not match",
            ),
            (
                (100, 3, 2, 10, 2),  # contours[:3] = (100, 3, 2)
                (90, 3, 2),  # wrong frames
                "Segmentation data shape does not match",
            ),
            (
                (100, 3, 2, 10, 2),  # contours[:3] = (100, 3, 2)
                (100, 3, 3),  # wrong contours
                "Segmentation data shape does not match",
            ),
            (
                (50, 2, 1, 8, 2),  # contours[:3] = (50, 2, 1)
                (60, 3, 2),  # all wrong
                "Segmentation data shape does not match",
            ),
        ],
        ids=[
            "animals_mismatch",
            "frames_mismatch",
            "contours_mismatch",
            "all_mismatch",
        ],
    )
    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_various_shape_mismatches(
        self,
        mock_h5py_file,
        mock_adjust_pose_version,
        contours_shape,
        flags_shape,
        expected_error,
    ):
        """Test various combinations of shape mismatches."""
        # Arrange
        pose_file = "test_pose.h5"
        seg_contours_matrix = np.random.randint(
            0, 50, size=contours_shape, dtype=np.int32
        )
        seg_external_flags = np.random.randint(0, 2, size=flags_shape, dtype=np.int32)

        existing_datasets = []
        mock_context = create_mock_h5_context(existing_datasets)
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Act & Assert
        with pytest.raises(InvalidPoseFileException, match=expected_error):
            write_seg_data(pose_file, seg_contours_matrix, seg_external_flags)


class TestWriteSegDataCompression:
    """Test compression settings for write_seg_data."""

    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_gzip_compression_applied(self, mock_h5py_file, mock_adjust_pose_version):
        """Test that gzip compression is applied to both datasets."""
        # Arrange
        pose_file = "test_pose.h5"
        seg_contours_matrix = np.random.randint(
            0, 100, size=(20, 1, 3, 5, 2), dtype=np.int32
        )
        seg_external_flags = np.random.randint(0, 2, size=(20, 1, 3), dtype=np.int32)

        existing_datasets = []
        mock_context = create_mock_h5_context(existing_datasets)
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Act
        write_seg_data(pose_file, seg_contours_matrix, seg_external_flags)

        # Assert
        # Check seg_data compression
        seg_data_info = mock_context.created_datasets["poseest/seg_data"]
        assert seg_data_info["kwargs"]["compression"] == "gzip"
        assert seg_data_info["kwargs"]["compression_opts"] == 9

        # Check seg_external_flag compression
        flag_info = mock_context.created_datasets["poseest/seg_external_flag"]
        assert flag_info["kwargs"]["compression"] == "gzip"
        assert flag_info["kwargs"]["compression_opts"] == 9


class TestWriteSegDataAttributes:
    """Test attribute handling for write_seg_data."""

    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_attributes_set_only_on_seg_data(
        self, mock_h5py_file, mock_adjust_pose_version
    ):
        """Test that attributes are only set on seg_data, not on seg_external_flag."""
        # Arrange
        pose_file = "test_pose.h5"
        seg_contours_matrix = np.random.randint(
            0, 80, size=(15, 2, 1, 6, 2), dtype=np.int32
        )
        seg_external_flags = np.random.randint(0, 2, size=(15, 2, 1), dtype=np.int32)
        config_str = "segmentation_config"
        model_str = "segmentation_model"

        existing_datasets = []
        mock_context = create_mock_h5_context(existing_datasets)
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Act
        write_seg_data(
            pose_file, seg_contours_matrix, seg_external_flags, config_str, model_str
        )

        # Assert
        # Check that seg_data has attributes
        seg_data_info = mock_context.created_datasets["poseest/seg_data"]
        seg_dataset = seg_data_info["dataset"]
        assert seg_dataset.attrs["config"] == config_str
        assert seg_dataset.attrs["model"] == model_str

        # Check that seg_external_flag does NOT have these attributes set
        flag_info = mock_context.created_datasets["poseest/seg_external_flag"]
        flag_dataset = flag_info["dataset"]
        # Attributes should be empty MockAttrs (no explicit setting)
        assert len(flag_dataset.attrs._data) == 0

    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_string_attributes_with_special_characters(
        self, mock_h5py_file, mock_adjust_pose_version
    ):
        """Test setting attributes with special characters."""
        # Arrange
        pose_file = "test_pose.h5"
        seg_contours_matrix = np.random.randint(
            0, 50, size=(10, 1, 2, 4, 2), dtype=np.int32
        )
        seg_external_flags = np.random.randint(0, 2, size=(10, 1, 2), dtype=np.int32)
        config_str = "config/with/slashes_and-dashes & symbols"
        model_str = "model:checkpoint@v1.0 (final)"

        existing_datasets = []
        mock_context = create_mock_h5_context(existing_datasets)
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Act
        write_seg_data(
            pose_file, seg_contours_matrix, seg_external_flags, config_str, model_str
        )

        # Assert
        seg_data_info = mock_context.created_datasets["poseest/seg_data"]
        seg_dataset = seg_data_info["dataset"]
        assert seg_dataset.attrs["config"] == config_str
        assert seg_dataset.attrs["model"] == model_str


class TestWriteSegDataVersionHandling:
    """Test version promotion handling for write_seg_data."""

    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_adjust_pose_version_called_when_not_skipped(
        self, mock_h5py_file, mock_adjust_pose_version
    ):
        """Test that adjust_pose_version is called when skip_matching=False."""
        # Arrange
        pose_file = "test_pose.h5"
        seg_contours_matrix = np.random.randint(
            0, 40, size=(30, 2, 2, 8, 2), dtype=np.int32
        )
        seg_external_flags = np.random.randint(0, 2, size=(30, 2, 2), dtype=np.int32)

        existing_datasets = []
        mock_context = create_mock_h5_context(existing_datasets)
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Act
        write_seg_data(
            pose_file, seg_contours_matrix, seg_external_flags, skip_matching=False
        )

        # Assert
        # Should call adjust_pose_version with version 6
        mock_adjust_pose_version.assert_called_once_with(pose_file, 6)

    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_adjust_pose_version_not_called_when_skipped(
        self, mock_h5py_file, mock_adjust_pose_version
    ):
        """Test that adjust_pose_version is not called when skip_matching=True."""
        # Arrange
        pose_file = "test_pose.h5"
        seg_contours_matrix = np.random.randint(
            0, 60, size=(25, 3, 1, 10, 2), dtype=np.int32
        )
        seg_external_flags = np.random.randint(0, 2, size=(25, 3, 1), dtype=np.int32)

        existing_datasets = []
        mock_context = create_mock_h5_context(existing_datasets)
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Act
        write_seg_data(
            pose_file, seg_contours_matrix, seg_external_flags, skip_matching=True
        )

        # Assert
        # Should not call adjust_pose_version
        mock_adjust_pose_version.assert_not_called()


class TestWriteSegDataEdgeCases:
    """Test edge cases for write_seg_data."""

    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_empty_data_arrays(self, mock_h5py_file, mock_adjust_pose_version):
        """Test handling of empty data arrays."""
        # Arrange
        pose_file = "test_pose.h5"
        seg_contours_matrix = np.array([], dtype=np.int32).reshape(0, 0, 0, 5, 2)
        seg_external_flags = np.array([], dtype=np.int32).reshape(0, 0, 0)

        existing_datasets = []
        mock_context = create_mock_h5_context(existing_datasets)
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Act
        write_seg_data(pose_file, seg_contours_matrix, seg_external_flags)

        # Assert
        # Should successfully create datasets even with empty data
        assert "poseest/seg_data" in mock_context.created_datasets
        assert "poseest/seg_external_flag" in mock_context.created_datasets

        seg_data_info = mock_context.created_datasets["poseest/seg_data"]
        flag_info = mock_context.created_datasets["poseest/seg_external_flag"]

        assert seg_data_info["data"].shape == (0, 0, 0, 5, 2)
        assert flag_info["data"].shape == (0, 0, 0)

    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_single_frame_data(self, mock_h5py_file, mock_adjust_pose_version):
        """Test handling of single frame data."""
        # Arrange
        pose_file = "test_pose.h5"
        seg_contours_matrix = np.random.randint(
            0, 30, size=(1, 2, 3, 6, 2), dtype=np.int32
        )  # Single frame
        seg_external_flags = np.random.randint(0, 2, size=(1, 2, 3), dtype=np.int32)

        existing_datasets = []
        mock_context = create_mock_h5_context(existing_datasets)
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Act
        write_seg_data(pose_file, seg_contours_matrix, seg_external_flags)

        # Assert
        seg_data_info = mock_context.created_datasets["poseest/seg_data"]
        flag_info = mock_context.created_datasets["poseest/seg_external_flag"]

        np.testing.assert_array_equal(seg_data_info["data"], seg_contours_matrix)
        np.testing.assert_array_equal(flag_info["data"], seg_external_flags)

    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_single_animal_data(self, mock_h5py_file, mock_adjust_pose_version):
        """Test handling of single animal data."""
        # Arrange
        pose_file = "test_pose.h5"
        seg_contours_matrix = np.random.randint(
            0, 40, size=(50, 1, 2, 8, 2), dtype=np.int32
        )  # Single animal
        seg_external_flags = np.random.randint(0, 2, size=(50, 1, 2), dtype=np.int32)

        existing_datasets = []
        mock_context = create_mock_h5_context(existing_datasets)
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Act
        write_seg_data(pose_file, seg_contours_matrix, seg_external_flags)

        # Assert
        assert "poseest/seg_data" in mock_context.created_datasets
        assert "poseest/seg_external_flag" in mock_context.created_datasets

        seg_data_info = mock_context.created_datasets["poseest/seg_data"]
        flag_info = mock_context.created_datasets["poseest/seg_external_flag"]

        assert seg_data_info["data"].shape == (50, 1, 2, 8, 2)
        assert flag_info["data"].shape == (50, 1, 2)

    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_large_contour_data(self, mock_h5py_file, mock_adjust_pose_version):
        """Test handling of large contour data."""
        # Arrange
        pose_file = "test_pose.h5"
        seg_contours_matrix = np.random.randint(
            0, 200, size=(100, 3, 5, 50, 2), dtype=np.int32
        )  # Large contours
        seg_external_flags = np.random.randint(0, 2, size=(100, 3, 5), dtype=np.int32)

        existing_datasets = []
        mock_context = create_mock_h5_context(existing_datasets)
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Act
        write_seg_data(pose_file, seg_contours_matrix, seg_external_flags)

        # Assert
        seg_data_info = mock_context.created_datasets["poseest/seg_data"]
        flag_info = mock_context.created_datasets["poseest/seg_external_flag"]

        np.testing.assert_array_equal(seg_data_info["data"], seg_contours_matrix)
        np.testing.assert_array_equal(flag_info["data"], seg_external_flags)

        # Should still use compression for large data
        assert seg_data_info["kwargs"]["compression"] == "gzip"
        assert flag_info["kwargs"]["compression"] == "gzip"


class TestWriteSegDataIntegration:
    """Integration-style tests for write_seg_data."""

    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_complete_workflow_with_realistic_data(
        self, mock_h5py_file, mock_adjust_pose_version
    ):
        """Test complete workflow with realistic segmentation data."""
        # Arrange
        pose_file = "realistic_seg.h5"
        num_frames = 200
        num_animals = 2
        num_contours = 3
        max_contour_length = 20

        # Create realistic segmentation data
        seg_contours_matrix = np.random.randint(
            -1,
            300,
            size=(num_frames, num_animals, num_contours, max_contour_length, 2),
            dtype=np.int32,
        )
        seg_external_flags = np.random.randint(
            0, 2, size=(num_frames, num_animals, num_contours), dtype=np.int32
        )

        config_str = "unet_segmentation_v3.yaml"
        model_str = "segmentation_checkpoint_epoch_150.pth"

        existing_datasets = []
        mock_context = create_mock_h5_context(existing_datasets)
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Act
        write_seg_data(
            pose_file, seg_contours_matrix, seg_external_flags, config_str, model_str
        )

        # Assert
        # Verify datasets were created correctly
        assert "poseest/seg_data" in mock_context.created_datasets
        assert "poseest/seg_external_flag" in mock_context.created_datasets

        seg_data_info = mock_context.created_datasets["poseest/seg_data"]
        flag_info = mock_context.created_datasets["poseest/seg_external_flag"]

        # Verify data integrity
        np.testing.assert_array_equal(seg_data_info["data"], seg_contours_matrix)
        np.testing.assert_array_equal(flag_info["data"], seg_external_flags)

        # Verify compression settings
        assert seg_data_info["kwargs"]["compression"] == "gzip"
        assert seg_data_info["kwargs"]["compression_opts"] == 9
        assert flag_info["kwargs"]["compression"] == "gzip"
        assert flag_info["kwargs"]["compression_opts"] == 9

        # Verify attributes
        seg_dataset = seg_data_info["dataset"]
        assert seg_dataset.attrs["config"] == config_str
        assert seg_dataset.attrs["model"] == model_str

        # Verify version promotion was called
        mock_adjust_pose_version.assert_called_once_with(pose_file, 6)

    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_workflow_with_dataset_replacement(
        self, mock_h5py_file, mock_adjust_pose_version
    ):
        """Test workflow where existing segmentation datasets are replaced."""
        # Arrange
        pose_file = "test_pose.h5"
        seg_contours_matrix = np.random.randint(
            0, 100, size=(75, 3, 2, 15, 2), dtype=np.int32
        )
        seg_external_flags = np.random.randint(0, 2, size=(75, 3, 2), dtype=np.int32)
        config_str = "updated_config"
        model_str = "updated_model"

        # Mock existing datasets that will be replaced
        existing_datasets = ["poseest/seg_data", "poseest/seg_external_flag"]
        mock_context = create_mock_h5_context(existing_datasets)
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Act
        write_seg_data(
            pose_file, seg_contours_matrix, seg_external_flags, config_str, model_str
        )

        # Assert
        # Should delete existing datasets
        assert "poseest/seg_data" in mock_context.deleted_datasets
        assert "poseest/seg_external_flag" in mock_context.deleted_datasets

        # Should create new datasets with correct data
        assert "poseest/seg_data" in mock_context.created_datasets
        assert "poseest/seg_external_flag" in mock_context.created_datasets

        seg_data_info = mock_context.created_datasets["poseest/seg_data"]
        flag_info = mock_context.created_datasets["poseest/seg_external_flag"]

        np.testing.assert_array_equal(seg_data_info["data"], seg_contours_matrix)
        np.testing.assert_array_equal(flag_info["data"], seg_external_flags)

        # Verify new attributes
        seg_dataset = seg_data_info["dataset"]
        assert seg_dataset.attrs["config"] == config_str
        assert seg_dataset.attrs["model"] == model_str

    @patch("mouse_tracking.utils.writers.adjust_pose_version")
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_workflow_with_topdown_skip_matching(
        self, mock_h5py_file, mock_adjust_pose_version
    ):
        """Test workflow with skip_matching=True (topdown scenario)."""
        # Arrange
        pose_file = "topdown_pose.h5"
        seg_contours_matrix = np.random.randint(
            0, 150, size=(100, 4, 1, 25, 2), dtype=np.int32
        )
        seg_external_flags = np.random.randint(0, 2, size=(100, 4, 1), dtype=np.int32)
        config_str = "topdown_config"
        model_str = "topdown_model"

        existing_datasets = []
        mock_context = create_mock_h5_context(existing_datasets)
        mock_h5py_file.return_value.__enter__.return_value = mock_context

        # Act
        write_seg_data(
            pose_file,
            seg_contours_matrix,
            seg_external_flags,
            config_str,
            model_str,
            skip_matching=True,
        )

        # Assert
        # Should create datasets normally
        assert "poseest/seg_data" in mock_context.created_datasets
        assert "poseest/seg_external_flag" in mock_context.created_datasets

        # Should set attributes normally
        seg_data_info = mock_context.created_datasets["poseest/seg_data"]
        seg_dataset = seg_data_info["dataset"]
        assert seg_dataset.attrs["config"] == config_str
        assert seg_dataset.attrs["model"] == model_str

        # Should NOT call adjust_pose_version
        mock_adjust_pose_version.assert_not_called()
