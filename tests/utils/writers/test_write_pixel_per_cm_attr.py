"""Tests for write_pixel_per_cm_attr function."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pytest

from mouse_tracking.utils.writers import write_pixel_per_cm_attr


def test_writes_pixel_per_cm_attributes_successfully():
    """Test writing pixel per cm attributes to a new file."""
    # Arrange
    px_per_cm = 0.1
    source = "corner_detection"

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
        pose_file = tmp_file.name

    try:
        with patch("mouse_tracking.utils.writers.h5py.File") as mock_h5_file:
            # Setup mock file structure
            mock_file = MagicMock()
            mock_h5_file.return_value.__enter__.return_value = mock_file
            mock_poseest = MagicMock()
            mock_attrs = MagicMock()
            mock_poseest.attrs = mock_attrs
            mock_file.__getitem__.return_value = mock_poseest

            # Act
            write_pixel_per_cm_attr(pose_file, px_per_cm, source)

            # Assert
            mock_h5_file.assert_called_once_with(pose_file, "a")
            assert (
                mock_file.__getitem__.call_count == 2
            )  # Called twice - once for each attribute
            mock_file.__getitem__.assert_any_call("poseest")
            mock_attrs.__setitem__.assert_any_call("cm_per_pixel", px_per_cm)
            mock_attrs.__setitem__.assert_any_call("cm_per_pixel_source", source)

    finally:
        if os.path.exists(pose_file):
            os.unlink(pose_file)


@pytest.mark.parametrize(
    "px_per_cm,source",
    [
        (0.1, "corner_detection"),
        (0.05, "default_alignment"),
        (0.2, "manual"),
        (0.08, "automated_calibration"),
        (1.0, "manually_set"),
        (0.001, "test_source"),
        (100.0, "high_resolution"),
    ],
)
def test_writes_various_values_and_sources(px_per_cm, source):
    """Test writing different pixel per cm values and sources."""
    # Arrange
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
        pose_file = tmp_file.name

    try:
        with patch("mouse_tracking.utils.writers.h5py.File") as mock_h5_file:
            mock_file = MagicMock()
            mock_h5_file.return_value.__enter__.return_value = mock_file
            mock_poseest = MagicMock()
            mock_attrs = MagicMock()
            mock_poseest.attrs = mock_attrs
            mock_file.__getitem__.return_value = mock_poseest

            # Act
            write_pixel_per_cm_attr(pose_file, px_per_cm, source)

            # Assert
            mock_attrs.__setitem__.assert_any_call("cm_per_pixel", px_per_cm)
            mock_attrs.__setitem__.assert_any_call("cm_per_pixel_source", source)

    finally:
        if os.path.exists(pose_file):
            os.unlink(pose_file)


def test_writes_with_float32_value():
    """Test writing with numpy float32 value."""
    # Arrange
    px_per_cm = np.float32(0.15)
    source = "test_source"

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
        pose_file = tmp_file.name

    try:
        with patch("mouse_tracking.utils.writers.h5py.File") as mock_h5_file:
            mock_file = MagicMock()
            mock_h5_file.return_value.__enter__.return_value = mock_file
            mock_poseest = MagicMock()
            mock_attrs = MagicMock()
            mock_poseest.attrs = mock_attrs
            mock_file.__getitem__.return_value = mock_poseest

            # Act
            write_pixel_per_cm_attr(pose_file, px_per_cm, source)

            # Assert
            mock_attrs.__setitem__.assert_any_call("cm_per_pixel", px_per_cm)
            mock_attrs.__setitem__.assert_any_call("cm_per_pixel_source", source)

    finally:
        if os.path.exists(pose_file):
            os.unlink(pose_file)


def test_writes_with_integer_value():
    """Test writing with integer value (should be converted to float)."""
    # Arrange
    px_per_cm = 1  # integer
    source = "test_source"

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
        pose_file = tmp_file.name

    try:
        with patch("mouse_tracking.utils.writers.h5py.File") as mock_h5_file:
            mock_file = MagicMock()
            mock_h5_file.return_value.__enter__.return_value = mock_file
            mock_poseest = MagicMock()
            mock_attrs = MagicMock()
            mock_poseest.attrs = mock_attrs
            mock_file.__getitem__.return_value = mock_poseest

            # Act
            write_pixel_per_cm_attr(pose_file, px_per_cm, source)

            # Assert
            mock_attrs.__setitem__.assert_any_call("cm_per_pixel", px_per_cm)
            mock_attrs.__setitem__.assert_any_call("cm_per_pixel_source", source)

    finally:
        if os.path.exists(pose_file):
            os.unlink(pose_file)


def test_overwrites_existing_attributes():
    """Test overwriting existing pixel per cm attributes."""
    # Arrange
    px_per_cm = 0.25
    source = "new_source"

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
        pose_file = tmp_file.name

    try:
        with patch("mouse_tracking.utils.writers.h5py.File") as mock_h5_file:
            mock_file = MagicMock()
            mock_h5_file.return_value.__enter__.return_value = mock_file
            mock_poseest = MagicMock()
            mock_attrs = MagicMock()
            mock_poseest.attrs = mock_attrs
            mock_file.__getitem__.return_value = mock_poseest

            # Act
            write_pixel_per_cm_attr(pose_file, px_per_cm, source)

            # Assert
            mock_attrs.__setitem__.assert_any_call("cm_per_pixel", px_per_cm)
            mock_attrs.__setitem__.assert_any_call("cm_per_pixel_source", source)

    finally:
        if os.path.exists(pose_file):
            os.unlink(pose_file)


def test_handles_empty_source_string():
    """Test writing with empty source string."""
    # Arrange
    px_per_cm = 0.1
    source = ""

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
        pose_file = tmp_file.name

    try:
        with patch("mouse_tracking.utils.writers.h5py.File") as mock_h5_file:
            mock_file = MagicMock()
            mock_h5_file.return_value.__enter__.return_value = mock_file
            mock_poseest = MagicMock()
            mock_attrs = MagicMock()
            mock_poseest.attrs = mock_attrs
            mock_file.__getitem__.return_value = mock_poseest

            # Act
            write_pixel_per_cm_attr(pose_file, px_per_cm, source)

            # Assert
            mock_attrs.__setitem__.assert_any_call("cm_per_pixel", px_per_cm)
            mock_attrs.__setitem__.assert_any_call("cm_per_pixel_source", source)

    finally:
        if os.path.exists(pose_file):
            os.unlink(pose_file)


def test_handles_unicode_source_string():
    """Test writing with unicode source string."""
    # Arrange
    px_per_cm = 0.1
    source = "来源测试"  # Unicode source

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
        pose_file = tmp_file.name

    try:
        with patch("mouse_tracking.utils.writers.h5py.File") as mock_h5_file:
            mock_file = MagicMock()
            mock_h5_file.return_value.__enter__.return_value = mock_file
            mock_poseest = MagicMock()
            mock_attrs = MagicMock()
            mock_poseest.attrs = mock_attrs
            mock_file.__getitem__.return_value = mock_poseest

            # Act
            write_pixel_per_cm_attr(pose_file, px_per_cm, source)

            # Assert
            mock_attrs.__setitem__.assert_any_call("cm_per_pixel", px_per_cm)
            mock_attrs.__setitem__.assert_any_call("cm_per_pixel_source", source)

    finally:
        if os.path.exists(pose_file):
            os.unlink(pose_file)


def test_handles_special_characters_in_source():
    """Test writing with special characters in source string."""
    # Arrange
    px_per_cm = 0.1
    source = "test/source with spaces & symbols!"

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
        pose_file = tmp_file.name

    try:
        with patch("mouse_tracking.utils.writers.h5py.File") as mock_h5_file:
            mock_file = MagicMock()
            mock_h5_file.return_value.__enter__.return_value = mock_file
            mock_poseest = MagicMock()
            mock_attrs = MagicMock()
            mock_poseest.attrs = mock_attrs
            mock_file.__getitem__.return_value = mock_poseest

            # Act
            write_pixel_per_cm_attr(pose_file, px_per_cm, source)

            # Assert
            mock_attrs.__setitem__.assert_any_call("cm_per_pixel", px_per_cm)
            mock_attrs.__setitem__.assert_any_call("cm_per_pixel_source", source)

    finally:
        if os.path.exists(pose_file):
            os.unlink(pose_file)


def test_handles_extreme_small_values():
    """Test writing with extremely small pixel per cm values."""
    # Arrange
    px_per_cm = 1e-10
    source = "microscopic"

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
        pose_file = tmp_file.name

    try:
        with patch("mouse_tracking.utils.writers.h5py.File") as mock_h5_file:
            mock_file = MagicMock()
            mock_h5_file.return_value.__enter__.return_value = mock_file
            mock_poseest = MagicMock()
            mock_attrs = MagicMock()
            mock_poseest.attrs = mock_attrs
            mock_file.__getitem__.return_value = mock_poseest

            # Act
            write_pixel_per_cm_attr(pose_file, px_per_cm, source)

            # Assert
            mock_attrs.__setitem__.assert_any_call("cm_per_pixel", px_per_cm)
            mock_attrs.__setitem__.assert_any_call("cm_per_pixel_source", source)

    finally:
        if os.path.exists(pose_file):
            os.unlink(pose_file)


def test_handles_extreme_large_values():
    """Test writing with extremely large pixel per cm values."""
    # Arrange
    px_per_cm = 1e10
    source = "massive"

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
        pose_file = tmp_file.name

    try:
        with patch("mouse_tracking.utils.writers.h5py.File") as mock_h5_file:
            mock_file = MagicMock()
            mock_h5_file.return_value.__enter__.return_value = mock_file
            mock_poseest = MagicMock()
            mock_attrs = MagicMock()
            mock_poseest.attrs = mock_attrs
            mock_file.__getitem__.return_value = mock_poseest

            # Act
            write_pixel_per_cm_attr(pose_file, px_per_cm, source)

            # Assert
            mock_attrs.__setitem__.assert_any_call("cm_per_pixel", px_per_cm)
            mock_attrs.__setitem__.assert_any_call("cm_per_pixel_source", source)

    finally:
        if os.path.exists(pose_file):
            os.unlink(pose_file)


def test_propagates_h5py_file_exceptions():
    """Test that HDF5 file exceptions are propagated correctly."""
    # Arrange
    px_per_cm = 0.1
    source = "test_source"
    pose_file = "nonexistent_file.h5"

    with patch("mouse_tracking.utils.writers.h5py.File") as mock_h5_file:
        mock_h5_file.side_effect = OSError("File not found")

        # Act & Assert
        with pytest.raises(OSError, match="File not found"):
            write_pixel_per_cm_attr(pose_file, px_per_cm, source)


def test_propagates_poseest_group_missing_exceptions():
    """Test that missing poseest group exceptions are propagated correctly."""
    # Arrange
    px_per_cm = 0.1
    source = "test_source"
    pose_file = "test_file.h5"

    with patch("mouse_tracking.utils.writers.h5py.File") as mock_h5_file:
        mock_file = MagicMock()
        mock_h5_file.return_value.__enter__.return_value = mock_file
        mock_file.__getitem__.side_effect = KeyError("poseest group not found")

        # Act & Assert
        with pytest.raises(KeyError, match="poseest group not found"):
            write_pixel_per_cm_attr(pose_file, px_per_cm, source)


def test_propagates_attribute_setting_exceptions():
    """Test that attribute setting exceptions are propagated correctly."""
    # Arrange
    px_per_cm = 0.1
    source = "test_source"
    pose_file = "test_file.h5"

    with patch("mouse_tracking.utils.writers.h5py.File") as mock_h5_file:
        mock_file = MagicMock()
        mock_h5_file.return_value.__enter__.return_value = mock_file
        mock_poseest = MagicMock()
        mock_attrs = MagicMock()
        mock_poseest.attrs = mock_attrs
        mock_file.__getitem__.return_value = mock_poseest
        mock_attrs.__setitem__.side_effect = RuntimeError("Attribute setting failed")

        # Act & Assert
        with pytest.raises(RuntimeError, match="Attribute setting failed"):
            write_pixel_per_cm_attr(pose_file, px_per_cm, source)


def test_function_signature_and_types():
    """Test that the function accepts correct types."""
    # Arrange
    pose_file = "test_file.h5"

    with patch("mouse_tracking.utils.writers.h5py.File") as mock_h5_file:
        mock_file = MagicMock()
        mock_h5_file.return_value.__enter__.return_value = mock_file
        mock_poseest = MagicMock()
        mock_attrs = MagicMock()
        mock_poseest.attrs = mock_attrs
        mock_file.__getitem__.return_value = mock_poseest

        # Act & Assert - Test different valid type combinations
        write_pixel_per_cm_attr(pose_file, 0.1, "string")  # float, str
        write_pixel_per_cm_attr(pose_file, 1, "string")  # int, str
        write_pixel_per_cm_attr(pose_file, np.float32(0.1), "string")  # np.float32, str


def test_integration_with_real_h5py_file():
    """Integration test with real HDF5 file operations."""
    # Arrange
    px_per_cm = 0.125
    source = "integration_test"

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
        pose_file = tmp_file.name

    try:
        # First create a minimal HDF5 file with poseest group
        with h5py.File(pose_file, "w") as f:
            f.create_group("poseest")

        # Act
        write_pixel_per_cm_attr(pose_file, px_per_cm, source)

        # Assert - Check that data was written correctly
        with h5py.File(pose_file, "r") as f:
            assert "poseest" in f
            assert "cm_per_pixel" in f["poseest"].attrs
            assert "cm_per_pixel_source" in f["poseest"].attrs
            assert f["poseest"].attrs["cm_per_pixel"] == px_per_cm
            assert f["poseest"].attrs["cm_per_pixel_source"] == source

    finally:
        if os.path.exists(pose_file):
            os.unlink(pose_file)


def test_integration_overwrites_existing_real_attributes():
    """Integration test that overwrites existing attributes in real HDF5 file."""
    # Arrange
    original_px_per_cm = 0.1
    original_source = "original"
    new_px_per_cm = 0.2
    new_source = "updated"

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
        pose_file = tmp_file.name

    try:
        # Create file with initial attributes
        with h5py.File(pose_file, "w") as f:
            poseest = f.create_group("poseest")
            poseest.attrs["cm_per_pixel"] = original_px_per_cm
            poseest.attrs["cm_per_pixel_source"] = original_source

        # Act - Overwrite with new values
        write_pixel_per_cm_attr(pose_file, new_px_per_cm, new_source)

        # Assert - Check that new values overwrote old values
        with h5py.File(pose_file, "r") as f:
            assert f["poseest"].attrs["cm_per_pixel"] == new_px_per_cm
            assert f["poseest"].attrs["cm_per_pixel_source"] == new_source

    finally:
        if os.path.exists(pose_file):
            os.unlink(pose_file)


def test_integration_with_existing_datasets():
    """Integration test with existing datasets in the file."""
    # Arrange
    px_per_cm = 0.1
    source = "test_with_datasets"

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
        pose_file = tmp_file.name

    try:
        # Create file with some existing datasets
        with h5py.File(pose_file, "w") as f:
            poseest = f.create_group("poseest")
            poseest.create_dataset("points", data=np.random.rand(10, 2, 12, 2))
            poseest.create_dataset("confidence", data=np.random.rand(10, 2, 12))

        # Act
        write_pixel_per_cm_attr(pose_file, px_per_cm, source)

        # Assert - Check that attributes were added without affecting datasets
        with h5py.File(pose_file, "r") as f:
            assert "points" in f["poseest"]
            assert "confidence" in f["poseest"]
            assert f["poseest"].attrs["cm_per_pixel"] == px_per_cm
            assert f["poseest"].attrs["cm_per_pixel_source"] == source

    finally:
        if os.path.exists(pose_file):
            os.unlink(pose_file)


def test_realistic_usage_patterns():
    """Test realistic usage patterns from the codebase."""
    # Arrange - Test patterns found in actual usage
    test_cases = [
        (0.1, "corner_detection"),
        (0.05, "default_alignment"),
        (0.08, "automated_calibration"),
        (0.1, "manual"),
        (0.2, "manually_set"),
    ]

    for px_per_cm, source in test_cases:
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
            pose_file = tmp_file.name

        try:
            # Create minimal file
            with h5py.File(pose_file, "w") as f:
                f.create_group("poseest")

            # Act
            write_pixel_per_cm_attr(pose_file, px_per_cm, source)

            # Assert
            with h5py.File(pose_file, "r") as f:
                assert f["poseest"].attrs["cm_per_pixel"] == px_per_cm
                assert f["poseest"].attrs["cm_per_pixel_source"] == source

        finally:
            if os.path.exists(pose_file):
                os.unlink(pose_file)
