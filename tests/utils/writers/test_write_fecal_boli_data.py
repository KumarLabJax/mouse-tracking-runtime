"""Tests for write_fecal_boli_data function."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pytest

from mouse_tracking.utils.writers import write_fecal_boli_data


def test_writes_fecal_boli_data_successfully():
    """Test writing fecal boli data to a new file."""
    # Arrange
    detections = np.array([[[10, 20], [30, 40]], [[50, 60], [0, 0]]], dtype=np.uint16)
    count_detections = np.array([2, 1], dtype=np.uint16)
    sample_frequency = 1800
    config_str = "fecal_boli_config"
    model_str = "fecal_boli_model"

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
        pose_file = tmp_file.name

    try:
        with patch("mouse_tracking.utils.writers.h5py.File") as mock_h5_file:
            # Setup mock file structure
            mock_file = MagicMock()
            mock_h5_file.return_value.__enter__.return_value = mock_file
            mock_file.__contains__.return_value = False  # No existing dynamic_objects
            mock_dataset = MagicMock()
            mock_file.create_dataset.return_value = mock_dataset
            mock_attrs = MagicMock()
            mock_file.__getitem__.return_value.attrs = mock_attrs

            # Act
            write_fecal_boli_data(
                pose_file,
                detections,
                count_detections,
                sample_frequency,
                config_str,
                model_str,
            )

            # Assert
            mock_h5_file.assert_called_once_with(pose_file, "a")
            mock_file.__contains__.assert_called_once_with("dynamic_objects")

            # Check datasets creation calls
            expected_sample_indices = (
                np.arange(len(detections)) * sample_frequency
            ).astype(np.uint32)
            assert mock_file.create_dataset.call_count == 3

            # Check individual calls by examining call arguments
            calls = mock_file.create_dataset.call_args_list

            # Check points dataset call
            points_call = calls[0]
            assert points_call[0][0] == "dynamic_objects/fecal_boli/points"
            np.testing.assert_array_equal(points_call[1]["data"], detections)

            # Check counts dataset call
            counts_call = calls[1]
            assert counts_call[0][0] == "dynamic_objects/fecal_boli/counts"
            np.testing.assert_array_equal(counts_call[1]["data"], count_detections)

            # Check sample_indices dataset call
            indices_call = calls[2]
            assert indices_call[0][0] == "dynamic_objects/fecal_boli/sample_indices"
            np.testing.assert_array_equal(
                indices_call[1]["data"], expected_sample_indices
            )

            # Check attributes
            mock_attrs.__setitem__.assert_any_call("config", config_str)
            mock_attrs.__setitem__.assert_any_call("model", model_str)

    finally:
        if os.path.exists(pose_file):
            os.unlink(pose_file)


def test_overwrites_existing_fecal_boli_data():
    """Test overwriting existing fecal boli data."""
    # Arrange
    detections = np.array([[[100, 200]]], dtype=np.uint16)
    count_detections = np.array([1], dtype=np.uint16)
    sample_frequency = 3600
    config_str = "new_config"
    model_str = "new_model"

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
        pose_file = tmp_file.name

    try:
        with patch("mouse_tracking.utils.writers.h5py.File") as mock_h5_file:
            # Setup mock file structure with existing data
            mock_file = MagicMock()
            mock_h5_file.return_value.__enter__.return_value = mock_file
            mock_dynamic_objects = MagicMock()
            mock_dataset = MagicMock()
            mock_attrs = MagicMock()

            # Mock the file behavior for checking dynamic objects
            mock_file.__contains__.side_effect = lambda x: x == "dynamic_objects"
            mock_file.__getitem__.side_effect = lambda x: (
                mock_dynamic_objects
                if x == "dynamic_objects"
                else type("MockGroup", (), {"attrs": mock_attrs})()
            )
            mock_dynamic_objects.__contains__.return_value = True  # fecal_boli exists
            mock_file.create_dataset.return_value = mock_dataset

            # Act
            write_fecal_boli_data(
                pose_file,
                detections,
                count_detections,
                sample_frequency,
                config_str,
                model_str,
            )

            # Assert
            mock_file.__contains__.assert_called_once_with("dynamic_objects")
            mock_dynamic_objects.__contains__.assert_called_once_with("fecal_boli")
            mock_file.__delitem__.assert_called_once_with("dynamic_objects/fecal_boli")
            mock_attrs.__setitem__.assert_any_call("config", config_str)
            mock_attrs.__setitem__.assert_any_call("model", model_str)

    finally:
        if os.path.exists(pose_file):
            os.unlink(pose_file)


def test_writes_with_default_empty_config_and_model():
    """Test writing fecal boli data with default empty config and model strings."""
    # Arrange
    detections = np.array([[[1, 2]]], dtype=np.uint16)
    count_detections = np.array([1], dtype=np.uint16)
    sample_frequency = 1800

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
        pose_file = tmp_file.name

    try:
        with patch("mouse_tracking.utils.writers.h5py.File") as mock_h5_file:
            mock_file = MagicMock()
            mock_h5_file.return_value.__enter__.return_value = mock_file
            mock_file.__contains__.return_value = False
            mock_dataset = MagicMock()
            mock_file.create_dataset.return_value = mock_dataset
            mock_attrs = MagicMock()
            mock_file.__getitem__.return_value.attrs = mock_attrs

            # Act
            write_fecal_boli_data(
                pose_file, detections, count_detections, sample_frequency
            )

            # Assert
            mock_attrs.__setitem__.assert_any_call("config", "")
            mock_attrs.__setitem__.assert_any_call("model", "")

    finally:
        if os.path.exists(pose_file):
            os.unlink(pose_file)


@pytest.mark.parametrize(
    "detections,count_detections,sample_frequency,config_str,model_str",
    [
        (
            np.array([[[10, 20], [30, 40]]], dtype=np.uint16),
            np.array([2], dtype=np.uint16),
            1800,
            "config1",
            "model1",
        ),
        (
            np.array([[[1, 2]], [[3, 4]], [[5, 6]]], dtype=np.uint16),
            np.array([1, 1, 1], dtype=np.uint16),
            3600,
            "config2",
            "model2",
        ),
        (
            np.array([[[0, 0]]], dtype=np.uint16),
            np.array([0], dtype=np.uint16),
            1,
            "minimal",
            "test",
        ),
        (
            np.array([], dtype=np.uint16).reshape(0, 0, 2),
            np.array([], dtype=np.uint16),
            7200,
            "",
            "",
        ),
        (
            np.array([[[100, 200], [300, 400], [500, 600]]], dtype=np.uint16),
            np.array([3], dtype=np.uint16),
            900,
            "large",
            "dataset",
        ),
    ],
)
def test_writes_various_data_types_and_shapes(
    detections, count_detections, sample_frequency, config_str, model_str
):
    """Test writing different data types and shapes."""
    # Arrange
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
        pose_file = tmp_file.name

    try:
        with patch("mouse_tracking.utils.writers.h5py.File") as mock_h5_file:
            mock_file = MagicMock()
            mock_h5_file.return_value.__enter__.return_value = mock_file
            mock_file.__contains__.return_value = False
            mock_dataset = MagicMock()
            mock_file.create_dataset.return_value = mock_dataset
            mock_attrs = MagicMock()
            mock_file.__getitem__.return_value.attrs = mock_attrs

            # Act
            write_fecal_boli_data(
                pose_file,
                detections,
                count_detections,
                sample_frequency,
                config_str,
                model_str,
            )

            # Assert
            expected_sample_indices = (
                np.arange(len(detections)) * sample_frequency
            ).astype(np.uint32)
            assert mock_file.create_dataset.call_count == 3

            # Check individual calls by examining call arguments
            calls = mock_file.create_dataset.call_args_list

            # Verify all three datasets are created with correct names and data
            call_names = [call[0][0] for call in calls]
            assert "dynamic_objects/fecal_boli/points" in call_names
            assert "dynamic_objects/fecal_boli/counts" in call_names
            assert "dynamic_objects/fecal_boli/sample_indices" in call_names

            # Check that data matches (find the right call for each)
            for call in calls:
                if call[0][0] == "dynamic_objects/fecal_boli/points":
                    np.testing.assert_array_equal(call[1]["data"], detections)
                elif call[0][0] == "dynamic_objects/fecal_boli/counts":
                    np.testing.assert_array_equal(call[1]["data"], count_detections)
                elif call[0][0] == "dynamic_objects/fecal_boli/sample_indices":
                    np.testing.assert_array_equal(
                        call[1]["data"], expected_sample_indices
                    )

            mock_attrs.__setitem__.assert_any_call("config", config_str)
            mock_attrs.__setitem__.assert_any_call("model", model_str)

    finally:
        if os.path.exists(pose_file):
            os.unlink(pose_file)


def test_calculates_sample_indices_correctly():
    """Test that sample indices are calculated correctly."""
    # Arrange
    detections = np.array([[[1, 2]], [[3, 4]], [[5, 6]], [[7, 8]]], dtype=np.uint16)
    count_detections = np.array([1, 1, 1, 1], dtype=np.uint16)
    sample_frequency = 1800

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
        pose_file = tmp_file.name

    try:
        with patch("mouse_tracking.utils.writers.h5py.File") as mock_h5_file:
            mock_file = MagicMock()
            mock_h5_file.return_value.__enter__.return_value = mock_file
            mock_file.__contains__.return_value = False
            mock_dataset = MagicMock()
            mock_file.create_dataset.return_value = mock_dataset
            mock_attrs = MagicMock()
            mock_file.__getitem__.return_value.attrs = mock_attrs

            # Act
            write_fecal_boli_data(
                pose_file, detections, count_detections, sample_frequency
            )

            # Assert
            expected_sample_indices = np.array([0, 1800, 3600, 5400], dtype=np.uint32)

            # Find the sample_indices call
            calls = mock_file.create_dataset.call_args_list
            sample_indices_call = None
            for call in calls:
                if call[0][0] == "dynamic_objects/fecal_boli/sample_indices":
                    sample_indices_call = call
                    break

            assert sample_indices_call is not None, "sample_indices dataset not created"
            np.testing.assert_array_equal(
                sample_indices_call[1]["data"], expected_sample_indices
            )

    finally:
        if os.path.exists(pose_file):
            os.unlink(pose_file)


def test_handles_unicode_strings_in_config_and_model():
    """Test handling unicode strings in config and model parameters."""
    # Arrange
    detections = np.array([[[1, 2]]], dtype=np.uint16)
    count_detections = np.array([1], dtype=np.uint16)
    sample_frequency = 1800
    config_str = "配置字符串"
    model_str = "模型字符串"

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
        pose_file = tmp_file.name

    try:
        with patch("mouse_tracking.utils.writers.h5py.File") as mock_h5_file:
            mock_file = MagicMock()
            mock_h5_file.return_value.__enter__.return_value = mock_file
            mock_file.__contains__.return_value = False
            mock_dataset = MagicMock()
            mock_file.create_dataset.return_value = mock_dataset
            mock_attrs = MagicMock()
            mock_file.__getitem__.return_value.attrs = mock_attrs

            # Act
            write_fecal_boli_data(
                pose_file,
                detections,
                count_detections,
                sample_frequency,
                config_str,
                model_str,
            )

            # Assert
            mock_attrs.__setitem__.assert_any_call("config", config_str)
            mock_attrs.__setitem__.assert_any_call("model", model_str)

    finally:
        if os.path.exists(pose_file):
            os.unlink(pose_file)


def test_handles_different_numpy_dtypes():
    """Test handling different numpy data types for detections and counts."""
    # Arrange - Test with different dtypes
    detections = np.array([[[10, 20]]], dtype=np.int32)  # Different dtype
    count_detections = np.array([1], dtype=np.int32)  # Different dtype
    sample_frequency = 1800

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
        pose_file = tmp_file.name

    try:
        with patch("mouse_tracking.utils.writers.h5py.File") as mock_h5_file:
            mock_file = MagicMock()
            mock_h5_file.return_value.__enter__.return_value = mock_file
            mock_file.__contains__.return_value = False
            mock_dataset = MagicMock()
            mock_file.create_dataset.return_value = mock_dataset
            mock_attrs = MagicMock()
            mock_file.__getitem__.return_value.attrs = mock_attrs

            # Act
            write_fecal_boli_data(
                pose_file, detections, count_detections, sample_frequency
            )

            # Assert - Should accept the data regardless of dtype
            assert mock_file.create_dataset.call_count == 3

            # Check that correct datasets were created
            calls = mock_file.create_dataset.call_args_list
            call_names = [call[0][0] for call in calls]
            assert "dynamic_objects/fecal_boli/points" in call_names
            assert "dynamic_objects/fecal_boli/counts" in call_names
            assert "dynamic_objects/fecal_boli/sample_indices" in call_names

    finally:
        if os.path.exists(pose_file):
            os.unlink(pose_file)


def test_propagates_h5py_file_exceptions():
    """Test that HDF5 file exceptions are propagated correctly."""
    # Arrange
    detections = np.array([[[1, 2]]], dtype=np.uint16)
    count_detections = np.array([1], dtype=np.uint16)
    sample_frequency = 1800
    pose_file = "nonexistent_file.h5"

    with patch("mouse_tracking.utils.writers.h5py.File") as mock_h5_file:
        mock_h5_file.side_effect = OSError("File not found")

        # Act & Assert
        with pytest.raises(OSError, match="File not found"):
            write_fecal_boli_data(
                pose_file, detections, count_detections, sample_frequency
            )


def test_propagates_dataset_creation_exceptions():
    """Test that dataset creation exceptions are propagated correctly."""
    # Arrange
    detections = np.array([[[1, 2]]], dtype=np.uint16)
    count_detections = np.array([1], dtype=np.uint16)
    sample_frequency = 1800
    pose_file = "test_file.h5"

    with patch("mouse_tracking.utils.writers.h5py.File") as mock_h5_file:
        mock_file = MagicMock()
        mock_h5_file.return_value.__enter__.return_value = mock_file
        mock_file.__contains__.return_value = False
        mock_file.create_dataset.side_effect = ValueError("Invalid dataset")

        # Act & Assert
        with pytest.raises(ValueError, match="Invalid dataset"):
            write_fecal_boli_data(
                pose_file, detections, count_detections, sample_frequency
            )


def test_propagates_attribute_setting_exceptions():
    """Test that attribute setting exceptions are propagated correctly."""
    # Arrange
    detections = np.array([[[1, 2]]], dtype=np.uint16)
    count_detections = np.array([1], dtype=np.uint16)
    sample_frequency = 1800
    pose_file = "test_file.h5"

    with patch("mouse_tracking.utils.writers.h5py.File") as mock_h5_file:
        mock_file = MagicMock()
        mock_h5_file.return_value.__enter__.return_value = mock_file
        mock_file.__contains__.return_value = False
        mock_dataset = MagicMock()
        mock_file.create_dataset.return_value = mock_dataset
        mock_attrs = MagicMock()
        mock_file.__getitem__.return_value.attrs = mock_attrs
        mock_attrs.__setitem__.side_effect = RuntimeError("Attribute setting failed")

        # Act & Assert
        with pytest.raises(RuntimeError, match="Attribute setting failed"):
            write_fecal_boli_data(
                pose_file, detections, count_detections, sample_frequency
            )


def test_function_signature_and_types():
    """Test that the function accepts correct types."""
    # Arrange
    pose_file = "test_file.h5"
    detections = np.array([[[1, 2]]], dtype=np.uint16)
    count_detections = np.array([1], dtype=np.uint16)

    with patch("mouse_tracking.utils.writers.h5py.File") as mock_h5_file:
        mock_file = MagicMock()
        mock_h5_file.return_value.__enter__.return_value = mock_file
        mock_file.__contains__.return_value = False
        mock_dataset = MagicMock()
        mock_file.create_dataset.return_value = mock_dataset
        mock_attrs = MagicMock()
        mock_file.__getitem__.return_value.attrs = mock_attrs

        # Act & Assert - Test different valid type combinations
        write_fecal_boli_data(
            pose_file, detections, count_detections, 1800
        )  # int sample_frequency
        write_fecal_boli_data(
            pose_file, detections, count_detections, 1800, "config", "model"
        )  # with strings


def test_dynamic_objects_group_exists_but_fecal_boli_does_not():
    """Test the case where dynamic_objects group exists but fecal_boli doesn't."""
    # Arrange
    detections = np.array([[[1, 2]]], dtype=np.uint16)
    count_detections = np.array([1], dtype=np.uint16)
    sample_frequency = 1800

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
        pose_file = tmp_file.name

    try:
        with patch("mouse_tracking.utils.writers.h5py.File") as mock_h5_file:
            mock_file = MagicMock()
            mock_h5_file.return_value.__enter__.return_value = mock_file
            mock_dynamic_objects = MagicMock()
            mock_dataset = MagicMock()
            mock_attrs = MagicMock()

            # Mock the file behavior for checking dynamic objects
            mock_file.__contains__.side_effect = lambda x: x == "dynamic_objects"
            mock_file.__getitem__.side_effect = lambda x: (
                mock_dynamic_objects
                if x == "dynamic_objects"
                else type("MockGroup", (), {"attrs": mock_attrs})()
            )
            mock_dynamic_objects.__contains__.return_value = (
                False  # fecal_boli doesn't exist
            )
            mock_file.create_dataset.return_value = mock_dataset

            # Act
            write_fecal_boli_data(
                pose_file, detections, count_detections, sample_frequency
            )

            # Assert
            mock_file.__contains__.assert_called_once_with("dynamic_objects")
            mock_dynamic_objects.__contains__.assert_called_once_with("fecal_boli")
            mock_file.__delitem__.assert_not_called()  # Should not delete non-existent object

    finally:
        if os.path.exists(pose_file):
            os.unlink(pose_file)


def test_integration_with_real_h5py_file():
    """Integration test with real HDF5 file operations."""
    # Arrange
    detections = np.array([[[10, 20], [30, 40]], [[50, 60], [0, 0]]], dtype=np.uint16)
    count_detections = np.array([2, 1], dtype=np.uint16)
    sample_frequency = 1800
    config_str = "test_config"
    model_str = "test_model"

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
        pose_file = tmp_file.name

    try:
        # Act
        write_fecal_boli_data(
            pose_file,
            detections,
            count_detections,
            sample_frequency,
            config_str,
            model_str,
        )

        # Assert - Check that data was written correctly
        with h5py.File(pose_file, "r") as f:
            assert "dynamic_objects/fecal_boli/points" in f
            assert "dynamic_objects/fecal_boli/counts" in f
            assert "dynamic_objects/fecal_boli/sample_indices" in f

            np.testing.assert_array_equal(
                f["dynamic_objects/fecal_boli/points"][:], detections
            )
            np.testing.assert_array_equal(
                f["dynamic_objects/fecal_boli/counts"][:], count_detections
            )

            expected_sample_indices = np.array([0, 1800], dtype=np.uint32)
            np.testing.assert_array_equal(
                f["dynamic_objects/fecal_boli/sample_indices"][:],
                expected_sample_indices,
            )

            assert f["dynamic_objects/fecal_boli"].attrs["config"] == config_str
            assert f["dynamic_objects/fecal_boli"].attrs["model"] == model_str

    finally:
        if os.path.exists(pose_file):
            os.unlink(pose_file)


def test_integration_overwrites_existing_real_data():
    """Integration test that overwrites existing data in real HDF5 file."""
    # Arrange
    original_detections = np.array([[[1, 2]], [[3, 4]]], dtype=np.uint16)
    original_count_detections = np.array([1, 1], dtype=np.uint16)
    new_detections = np.array([[[10, 20]]], dtype=np.uint16)
    new_count_detections = np.array([1], dtype=np.uint16)
    sample_frequency = 3600

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
        pose_file = tmp_file.name

    try:
        # First write original data
        write_fecal_boli_data(
            pose_file,
            original_detections,
            original_count_detections,
            1800,
            "config1",
            "model1",
        )

        # Then overwrite with new data
        write_fecal_boli_data(
            pose_file,
            new_detections,
            new_count_detections,
            sample_frequency,
            "config2",
            "model2",
        )

        # Assert - Check that new data overwrote old data
        with h5py.File(pose_file, "r") as f:
            np.testing.assert_array_equal(
                f["dynamic_objects/fecal_boli/points"][:], new_detections
            )
            np.testing.assert_array_equal(
                f["dynamic_objects/fecal_boli/counts"][:], new_count_detections
            )

            expected_sample_indices = np.array([0], dtype=np.uint32)
            np.testing.assert_array_equal(
                f["dynamic_objects/fecal_boli/sample_indices"][:],
                expected_sample_indices,
            )

            assert f["dynamic_objects/fecal_boli"].attrs["config"] == "config2"
            assert f["dynamic_objects/fecal_boli"].attrs["model"] == "model2"

    finally:
        if os.path.exists(pose_file):
            os.unlink(pose_file)


def test_realistic_usage_patterns():
    """Test realistic usage patterns from the codebase."""
    # Arrange - Test patterns found in actual usage
    test_cases = [
        (
            np.array([[[100, 200], [300, 400]]], dtype=np.uint16),
            np.array([2], dtype=np.uint16),
            1800,
            "fecal-boli",
            "checkpoint-100",
        ),
        (
            np.array([[[50, 60]], [[70, 80]], [[90, 100]]], dtype=np.uint16),
            np.array([1, 1, 1], dtype=np.uint16),
            3600,
            "fecal_boli_v2",
            "epoch_200",
        ),
        (
            np.array([], dtype=np.uint16).reshape(0, 0, 2),
            np.array([], dtype=np.uint16),
            1800,
            "",
            "",
        ),
    ]

    for (
        detections,
        count_detections,
        sample_frequency,
        config_str,
        model_str,
    ) in test_cases:
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
            pose_file = tmp_file.name

        try:
            # Act
            write_fecal_boli_data(
                pose_file,
                detections,
                count_detections,
                sample_frequency,
                config_str,
                model_str,
            )

            # Assert
            with h5py.File(pose_file, "r") as f:
                np.testing.assert_array_equal(
                    f["dynamic_objects/fecal_boli/points"][:], detections
                )
                np.testing.assert_array_equal(
                    f["dynamic_objects/fecal_boli/counts"][:], count_detections
                )
                assert f["dynamic_objects/fecal_boli"].attrs["config"] == config_str
                assert f["dynamic_objects/fecal_boli"].attrs["model"] == model_str

        finally:
            if os.path.exists(pose_file):
                os.unlink(pose_file)
