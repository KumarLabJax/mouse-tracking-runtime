"""Tests for write_static_object_data function."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pytest

from mouse_tracking.utils.writers import write_static_object_data


class TestWriteStaticObjectData:
    """Test class for write_static_object_data function."""


def test_writes_new_static_object_data_successfully():
    """Test writing static object data to a new file."""
    # Arrange
    test_data = np.array([[10, 20], [30, 40]], dtype=np.float32)
    object_name = "test_object"
    config_str = "test_config"
    model_str = "test_model"

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
        pose_file = tmp_file.name

    try:
        # Mock h5py.File and adjust_pose_version
        with (
            patch("mouse_tracking.utils.writers.h5py.File") as mock_h5_file,
            patch(
                "mouse_tracking.utils.writers.adjust_pose_version"
            ) as mock_adjust_version,
        ):
            # Setup mock file structure
            mock_file = MagicMock()
            mock_h5_file.return_value.__enter__.return_value = mock_file
            mock_file.__contains__.return_value = False  # No existing static_objects
            mock_dataset = MagicMock()
            mock_file.create_dataset.return_value = mock_dataset
            mock_attrs = MagicMock()
            mock_file.__getitem__.return_value = mock_dataset
            mock_dataset.attrs = mock_attrs

            # Act
            write_static_object_data(
                pose_file, test_data, object_name, config_str, model_str
            )

            # Assert
            mock_h5_file.assert_called_once_with(pose_file, "a")
            mock_file.__contains__.assert_called_once_with("static_objects")
            mock_file.create_dataset.assert_called_once_with(
                f"static_objects/{object_name}", data=test_data
            )
            mock_attrs.__setitem__.assert_any_call("config", config_str)
            mock_attrs.__setitem__.assert_any_call("model", model_str)
            mock_adjust_version.assert_called_once_with(pose_file, 5)

    finally:
        if os.path.exists(pose_file):
            os.unlink(pose_file)


def test_overwrites_existing_static_object_data():
    """Test overwriting existing static object data."""
    # Arrange
    test_data = np.array([[50, 60], [70, 80]], dtype=np.float32)
    object_name = "existing_object"
    config_str = "new_config"
    model_str = "new_model"

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
        pose_file = tmp_file.name

    try:
        with (
            patch("mouse_tracking.utils.writers.h5py.File") as mock_h5_file,
            patch(
                "mouse_tracking.utils.writers.adjust_pose_version"
            ) as mock_adjust_version,
        ):
            # Setup mock file structure with existing data
            mock_file = MagicMock()
            mock_h5_file.return_value.__enter__.return_value = mock_file
            mock_static_objects = MagicMock()
            mock_dataset = MagicMock()
            mock_attrs = MagicMock()
            mock_dataset.attrs = mock_attrs

            # Mock the file behavior for checking static objects
            mock_file.__contains__.side_effect = lambda x: x == "static_objects"
            mock_file.__getitem__.side_effect = (
                lambda x: mock_static_objects if x == "static_objects" else mock_dataset
            )
            mock_static_objects.__contains__.return_value = True  # Object exists
            mock_file.create_dataset.return_value = mock_dataset

            # Act
            write_static_object_data(
                pose_file, test_data, object_name, config_str, model_str
            )

            # Assert
            mock_h5_file.assert_called_once_with(pose_file, "a")
            mock_file.__contains__.assert_called_once_with("static_objects")
            mock_static_objects.__contains__.assert_called_once_with(object_name)
            mock_file.__delitem__.assert_called_once_with(
                f"static_objects/{object_name}"
            )
            mock_file.create_dataset.assert_called_once_with(
                f"static_objects/{object_name}", data=test_data
            )
            mock_attrs.__setitem__.assert_any_call("config", config_str)
            mock_attrs.__setitem__.assert_any_call("model", model_str)
            mock_adjust_version.assert_called_once_with(pose_file, 5)

    finally:
        if os.path.exists(pose_file):
            os.unlink(pose_file)


def test_writes_with_default_empty_config_and_model():
    """Test writing static object data with default empty config and model strings."""
    # Arrange
    test_data = np.array([[1, 2]], dtype=np.float32)
    object_name = "minimal_object"

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
        pose_file = tmp_file.name

    try:
        with (
            patch("mouse_tracking.utils.writers.h5py.File") as mock_h5_file,
            patch(
                "mouse_tracking.utils.writers.adjust_pose_version"
            ) as mock_adjust_version,
        ):
            mock_file = MagicMock()
            mock_h5_file.return_value.__enter__.return_value = mock_file
            mock_file.__contains__.return_value = False
            mock_dataset = MagicMock()
            mock_file.create_dataset.return_value = mock_dataset
            mock_attrs = MagicMock()
            mock_dataset.attrs = mock_attrs
            mock_file.__getitem__.return_value = mock_dataset

            # Act
            write_static_object_data(pose_file, test_data, object_name)

            # Assert
            mock_attrs.__setitem__.assert_any_call("config", "")
            mock_attrs.__setitem__.assert_any_call("model", "")
            mock_adjust_version.assert_called_once_with(pose_file, 5)

    finally:
        if os.path.exists(pose_file):
            os.unlink(pose_file)


@pytest.mark.parametrize(
    "test_data,object_name,config_str,model_str",
    [
        (
            np.array([[10, 20], [30, 40]], dtype=np.uint16),
            "corners",
            "corner_model",
            "v1.0",
        ),
        (
            np.array([[1.5, 2.5]], dtype=np.float32),
            "lixit",
            "lixit_detection",
            "checkpoint_123",
        ),
        (
            np.array([[100, 200], [300, 400], [500, 600]], dtype=np.int32),
            "food_hopper",
            "food_model",
            "latest",
        ),
        (np.array([]), "empty_object", "", ""),
        (
            np.array([[[1, 2], [3, 4]]], dtype=np.float64),
            "3d_object",
            "3d_config",
            "3d_model",
        ),
    ],
)
def test_writes_various_data_types_and_shapes(
    test_data, object_name, config_str, model_str
):
    """Test writing different data types and shapes."""
    # Arrange
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
        pose_file = tmp_file.name

    try:
        with (
            patch("mouse_tracking.utils.writers.h5py.File") as mock_h5_file,
            patch(
                "mouse_tracking.utils.writers.adjust_pose_version"
            ) as mock_adjust_version,
        ):
            mock_file = MagicMock()
            mock_h5_file.return_value.__enter__.return_value = mock_file
            mock_file.__contains__.return_value = False
            mock_dataset = MagicMock()
            mock_file.create_dataset.return_value = mock_dataset
            mock_attrs = MagicMock()
            mock_dataset.attrs = mock_attrs
            mock_file.__getitem__.return_value = mock_dataset

            # Act
            write_static_object_data(
                pose_file, test_data, object_name, config_str, model_str
            )

            # Assert
            mock_file.create_dataset.assert_called_once_with(
                f"static_objects/{object_name}", data=test_data
            )
            mock_attrs.__setitem__.assert_any_call("config", config_str)
            mock_attrs.__setitem__.assert_any_call("model", model_str)
            mock_adjust_version.assert_called_once_with(pose_file, 5)

    finally:
        if os.path.exists(pose_file):
            os.unlink(pose_file)


def test_handles_special_characters_in_object_name():
    """Test handling object names with special characters."""
    # Arrange
    test_data = np.array([[1, 2]], dtype=np.float32)
    object_name = "object_with_spaces and/slashes"

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
        pose_file = tmp_file.name

    try:
        with (
            patch("mouse_tracking.utils.writers.h5py.File") as mock_h5_file,
            patch(
                "mouse_tracking.utils.writers.adjust_pose_version"
            ) as mock_adjust_version,
        ):
            mock_file = MagicMock()
            mock_h5_file.return_value.__enter__.return_value = mock_file
            mock_file.__contains__.return_value = False
            mock_dataset = MagicMock()
            mock_file.create_dataset.return_value = mock_dataset
            mock_attrs = MagicMock()
            mock_dataset.attrs = mock_attrs
            mock_file.__getitem__.return_value = mock_dataset

            # Act
            write_static_object_data(pose_file, test_data, object_name)

            # Assert
            mock_file.create_dataset.assert_called_once_with(
                f"static_objects/{object_name}", data=test_data
            )
            mock_adjust_version.assert_called_once_with(pose_file, 5)

    finally:
        if os.path.exists(pose_file):
            os.unlink(pose_file)


def test_handles_unicode_strings_in_config_and_model():
    """Test handling unicode strings in config and model parameters."""
    # Arrange
    test_data = np.array([[1, 2]], dtype=np.float32)
    object_name = "unicode_test"
    config_str = "配置字符串"
    model_str = "模型字符串"

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
        pose_file = tmp_file.name

    try:
        with (
            patch("mouse_tracking.utils.writers.h5py.File") as mock_h5_file,
            patch(
                "mouse_tracking.utils.writers.adjust_pose_version"
            ) as mock_adjust_version,
        ):
            mock_file = MagicMock()
            mock_h5_file.return_value.__enter__.return_value = mock_file
            mock_file.__contains__.return_value = False
            mock_dataset = MagicMock()
            mock_file.create_dataset.return_value = mock_dataset
            mock_attrs = MagicMock()
            mock_dataset.attrs = mock_attrs
            mock_file.__getitem__.return_value = mock_dataset

            # Act
            write_static_object_data(
                pose_file, test_data, object_name, config_str, model_str
            )

            # Assert
            mock_attrs.__setitem__.assert_any_call("config", config_str)
            mock_attrs.__setitem__.assert_any_call("model", model_str)
            mock_adjust_version.assert_called_once_with(pose_file, 5)

    finally:
        if os.path.exists(pose_file):
            os.unlink(pose_file)


def test_propagates_h5py_file_exceptions():
    """Test that HDF5 file exceptions are propagated correctly."""
    # Arrange
    test_data = np.array([[1, 2]], dtype=np.float32)
    object_name = "test_object"
    pose_file = "nonexistent_file.h5"

    with patch("mouse_tracking.utils.writers.h5py.File") as mock_h5_file:
        mock_h5_file.side_effect = OSError("File not found")

        # Act & Assert
        with pytest.raises(OSError, match="File not found"):
            write_static_object_data(pose_file, test_data, object_name)


def test_propagates_h5py_dataset_creation_exceptions():
    """Test that HDF5 dataset creation exceptions are propagated correctly."""
    # Arrange
    test_data = np.array([[1, 2]], dtype=np.float32)
    object_name = "test_object"
    pose_file = "test_file.h5"

    with patch("mouse_tracking.utils.writers.h5py.File") as mock_h5_file:
        mock_file = MagicMock()
        mock_h5_file.return_value.__enter__.return_value = mock_file
        mock_file.__contains__.return_value = False
        mock_file.create_dataset.side_effect = ValueError("Invalid dataset")

        # Act & Assert
        with pytest.raises(ValueError, match="Invalid dataset"):
            write_static_object_data(pose_file, test_data, object_name)


def test_propagates_adjust_pose_version_exceptions():
    """Test that adjust_pose_version exceptions are propagated correctly."""
    # Arrange
    test_data = np.array([[1, 2]], dtype=np.float32)
    object_name = "test_object"
    pose_file = "test_file.h5"

    with (
        patch("mouse_tracking.utils.writers.h5py.File") as mock_h5_file,
        patch(
            "mouse_tracking.utils.writers.adjust_pose_version"
        ) as mock_adjust_version,
    ):
        mock_file = MagicMock()
        mock_h5_file.return_value.__enter__.return_value = mock_file
        mock_file.__contains__.return_value = False
        mock_dataset = MagicMock()
        mock_file.create_dataset.return_value = mock_dataset
        mock_attrs = MagicMock()
        mock_dataset.attrs = mock_attrs
        mock_adjust_version.side_effect = RuntimeError("Version adjustment failed")

        # Act & Assert
        with pytest.raises(RuntimeError, match="Version adjustment failed"):
            write_static_object_data(pose_file, test_data, object_name)


def test_function_signature_and_defaults():
    """Test that the function has the correct signature and default values."""
    # Arrange
    test_data = np.array([[1, 2]], dtype=np.float32)
    object_name = "test_object"
    pose_file = "test_file.h5"

    with (
        patch("mouse_tracking.utils.writers.h5py.File") as mock_h5_file,
        patch(
            "mouse_tracking.utils.writers.adjust_pose_version"
        ) as mock_adjust_version,
    ):
        mock_file = MagicMock()
        mock_h5_file.return_value.__enter__.return_value = mock_file
        mock_file.__contains__.return_value = False
        mock_dataset = MagicMock()
        mock_file.create_dataset.return_value = mock_dataset
        mock_attrs = MagicMock()
        mock_dataset.attrs = mock_attrs
        mock_file.__getitem__.return_value = mock_dataset

        # Act - Test calling with positional args only
        write_static_object_data(pose_file, test_data, object_name)

        # Assert
        mock_attrs.__setitem__.assert_any_call("config", "")
        mock_attrs.__setitem__.assert_any_call("model", "")
        mock_adjust_version.assert_called_once_with(pose_file, 5)


def test_static_objects_group_exists_but_object_does_not():
    """Test the case where static_objects group exists but the specific object doesn't."""
    # Arrange
    test_data = np.array([[1, 2]], dtype=np.float32)
    object_name = "new_object"

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
        pose_file = tmp_file.name

    try:
        with (
            patch("mouse_tracking.utils.writers.h5py.File") as mock_h5_file,
            patch(
                "mouse_tracking.utils.writers.adjust_pose_version"
            ) as mock_adjust_version,
        ):
            mock_file = MagicMock()
            mock_h5_file.return_value.__enter__.return_value = mock_file
            mock_static_objects = MagicMock()
            mock_dataset = MagicMock()
            mock_attrs = MagicMock()
            mock_dataset.attrs = mock_attrs

            # Mock the file behavior for checking static objects
            mock_file.__contains__.side_effect = lambda x: x == "static_objects"
            mock_file.__getitem__.side_effect = (
                lambda x: mock_static_objects if x == "static_objects" else mock_dataset
            )
            mock_static_objects.__contains__.return_value = (
                False  # Object doesn't exist
            )
            mock_file.create_dataset.return_value = mock_dataset

            # Act
            write_static_object_data(pose_file, test_data, object_name)

            # Assert
            mock_file.__contains__.assert_called_once_with("static_objects")
            mock_static_objects.__contains__.assert_called_once_with(object_name)
            mock_file.__delitem__.assert_not_called()  # Should not delete non-existent object
            mock_file.create_dataset.assert_called_once_with(
                f"static_objects/{object_name}", data=test_data
            )
            mock_adjust_version.assert_called_once_with(pose_file, 5)

    finally:
        if os.path.exists(pose_file):
            os.unlink(pose_file)


def test_integration_with_real_h5py_file():
    """Integration test with real HDF5 file operations."""
    # Arrange
    test_data = np.array([[10, 20], [30, 40]], dtype=np.float32)
    object_name = "corners"
    config_str = "test_config"
    model_str = "test_model"

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
        pose_file = tmp_file.name

    try:
        with patch(
            "mouse_tracking.utils.writers.adjust_pose_version"
        ) as mock_adjust_version:
            # Act
            write_static_object_data(
                pose_file, test_data, object_name, config_str, model_str
            )

            # Assert - Check that data was written correctly
            with h5py.File(pose_file, "r") as f:
                assert f"static_objects/{object_name}" in f
                stored_data = f[f"static_objects/{object_name}"][:]
                np.testing.assert_array_equal(stored_data, test_data)
                assert f[f"static_objects/{object_name}"].attrs["config"] == config_str
                assert f[f"static_objects/{object_name}"].attrs["model"] == model_str

            mock_adjust_version.assert_called_once_with(pose_file, 5)

    finally:
        if os.path.exists(pose_file):
            os.unlink(pose_file)


def test_integration_overwrites_existing_real_data():
    """Integration test that overwrites existing data in real HDF5 file."""
    # Arrange
    original_data = np.array([[1, 2], [3, 4]], dtype=np.float32)
    new_data = np.array([[10, 20], [30, 40]], dtype=np.float32)
    object_name = "test_object"

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
        pose_file = tmp_file.name

    try:
        with patch(
            "mouse_tracking.utils.writers.adjust_pose_version"
        ) as mock_adjust_version:
            # First write original data
            write_static_object_data(
                pose_file, original_data, object_name, "config1", "model1"
            )

            # Then overwrite with new data
            write_static_object_data(
                pose_file, new_data, object_name, "config2", "model2"
            )

            # Assert - Check that new data overwrote old data
            with h5py.File(pose_file, "r") as f:
                stored_data = f[f"static_objects/{object_name}"][:]
                np.testing.assert_array_equal(stored_data, new_data)
                assert f[f"static_objects/{object_name}"].attrs["config"] == "config2"
                assert f[f"static_objects/{object_name}"].attrs["model"] == "model2"

            assert mock_adjust_version.call_count == 2

    finally:
        if os.path.exists(pose_file):
            os.unlink(pose_file)
